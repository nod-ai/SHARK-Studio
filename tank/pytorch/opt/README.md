- [Model fixes](#model-fixes)
  * [Error 1](#error-1)
    + [Location](#location)
    + [Fix](#fix)
  * [Error 2](#error-2)
    + [Location](#location-1)
    + [Fix](#fix-1)
  * [Error 3](#error-3)
    + [Location](#location-2)
    + [Fix](#fix-2)
  * [Error 4](#error-4)
    + [Location](#location-3)
    + [Fix](#fix-3)
- [Bigger problems](#bigger-problems)


# Model fixes

## Error 1
```mlir
Exception: 
Lowering TorchScript IR -> Torch Backend IR failed with the following diagnostics:
error: 'func.call' op operand type mismatch: expected operand type '!torch.float', but provided '!torch.number' for operand number 0
note: see current operation: %1025 = "func.call"(%130, %1021, %1022, %1023, %1024) {callee = @__torch_mlir_shape_fn.aten.arange} : (!torch.number, !torch.optional<int>, !torch.optional<int>, !torch.optional<Device>, !torch.optional<bool>) -> !torch.list<int>
```

### Location

```python
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    ...
    mask_cond = torch.arange(mask.size(-1))
```

### Fix

```python
    mask_cond = torch.arange(mask.size(-1))
```

## Error 2

```mlir
Exception: 
Lowering TorchScript IR -> Torch Backend IR failed with the following diagnostics:
error: 'func.call' op operand type mismatch: expected operand type '!torch.float', but provided '!torch.number' for operand number 1
note: see current operation: %1023 = "func.call"(%126, %125, %1019, %1020, %1021, %1022) {callee = @__torch_mlir_shape_fn.aten.full} : (!torch.list<int>, !torch.number, !torch.optional<int>, !torch.optional<int>, !torch.optional<Device>, !torch.optional<bool>) -> !torch.list<int>
```

### Location

```python
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    ...
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")))
```

### Fix

```python
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")))
```

## Error 3

```mlir
Exception: 
Lowering TorchScript IR -> Torch Backend IR failed with the following diagnostics:
error: failed to legalize operation 'torch.aten.to.dtype_layout' that was explicitly marked illegal
note: see current operation: %141 = "torch.aten.to.dtype_layout"(%140, %83, %101, %117, %96, %98, %98, %96) : (!torch.vtensor<[1,1,7,7],f32>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.bool, !torch.bool, !torch.none) -> !torch.vtensor<[1,1,7,7],f32>
```

This one is confusing - the error is about `dtype_layout` but the source info points to a line with a `.to(device)`.
So there are two fixes.

### Location

```python
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    ...
    mask = mask.to(dtype)

...
    
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    ... 
    combined_attention_mask = _make_causal_mask(
        input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
    ).to(inputs_embeds.device) # <-------------------- here
```

### Fix

```python
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    ...
    #mask = mask.to(dtype)

...

def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    ...
    combined_attention_mask = _make_causal_mask(
        input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
    )#.to(inputs_embeds.device)
```

and for good measure eliminate the other `.to(dtype)` things

```python
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    ...
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)#.to(dtype)
    ...
    # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    return inverted_mask.masked_fill(inverted_mask, torch.finfo(dtype).min)
```

## Error 4

```mlir
Exception: 
Lowering TorchScript IR -> Torch Backend IR failed with the following diagnostics:
error: unsupported byte, char or bool type for convertScalarToDtype 'f32'(scalar type) -> 'i1'(dtype)
error: failed to legalize operation 'torch.aten.to.dtype' that was explicitly marked illegal
note: see current operation: %641 = "torch.aten.to.dtype"(%640, %97, %112, %112, %110) : (!torch.vtensor<[1,1,7,7],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none) -> !torch.vtensor<[1,1,7,7],i1>
```

This one is also confusing - the error is again about `dtype` but the source info points to a line with a no `dtype` casts.

### Location

```python
def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
```

The actual problem is that `bsz` is sometimes a `torch.Tensor`. Looking to the caller we find

```python
class OPTAttention(nn.Module):
    ...
    def forward(self):
        ...
        bsz, tgt_len, _ = hidden_states.size()
```

which mysteriously sometimes returns a tensor (I don't know why/how).

### Fix

```python
class OPTAttention(nn.Module):
    ...
    def forward(self):
        ...
        bsz, tgt_len, _ = map(int, hidden_states.size())
```

and for good measure in another place where `.size()` is called

```python
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    ...
    # bsz, src_len = mask.size()
    bsz, src_len = map(int, mask.size())
```

# Bigger problems

After the above you should be able to lower to torch dialect:

```python
module = torch_mlir.compile(
    model,
    (input_ids, attention_mask),
    output_type=torch_mlir.OutputType.TORCH,
    use_tracing=True,
)
```

but not all the way down to `linalg`; note that lowering to TOSA is dead in the water because of the `torch.Long` tensors which TOSA doesn't support (`error: Integers with widths greater than 32 are not supported
`).

The problem with lowering further involves the `torch-maximize-value-semantics` pass; if you lower to torch dialect and then use `torch-mlir-opt -pass-pipeline=torch-backend-to-linalg-on-tensors-backend-pipeline` what you find is that there are various `torch.aten.view`s that are getting passed `torch.tensor`s, which isn't supported.
The reason for this is that those `torch.tensor`s aren't "maximized" away by the `torch-maximize-value-semantics` pass, which currently doesn't support tuples that contain views; if you look at the torch dialect IR you'll see at the bottom

```mlir
    %852 = torch.prim.TupleConstruct %134, %139 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %853 = torch.prim.TupleConstruct %199, %204 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %854 = torch.prim.TupleConstruct %259, %264 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %855 = torch.prim.TupleConstruct %319, %324 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %856 = torch.prim.TupleConstruct %379, %384 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %857 = torch.prim.TupleConstruct %439, %444 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %858 = torch.prim.TupleConstruct %499, %504 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %859 = torch.prim.TupleConstruct %559, %564 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %860 = torch.prim.TupleConstruct %619, %624 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %861 = torch.prim.TupleConstruct %679, %684 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %862 = torch.prim.TupleConstruct %739, %744 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %863 = torch.prim.TupleConstruct %799, %804 : !torch.tensor<[1,12,7,64],f32>, !torch.tensor<[1,12,7,64],f32> -> !torch.tuple<tensor<[1,12,7,64],f32>, tensor<[1,12,7,64],f32>> loc(#loc0)
    %864 = torch.prim.TupleConstruct %852, %853, %854, %855, %856, %857, %858, %859, %860, %861, %862, %863 ....
    %865 = torch.prim.TupleConstruct %851, %864 
```

These are from the way the model passes returns around from all of the submodules:

```python
class OPTDecoderLayer(nn.Module):
    ...
    def forward(self):
        ...
        
        if not self.do_layer_norm_before:
              hidden_states = self.final_layer_norm(hidden_states)

          outputs = (hidden_states,)

          if output_attentions:
              outputs += (self_attn_weights,)

          if use_cache:
              outputs += (present_key_value,)

          return outputs

class OPTDecoder(nn.Module):
    ...
    def forward(self):
        ...
        
                    if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

class OPTForCausalLM(nn.Module):
    ...
    def forward(self):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            ...
        )

        logits = self.lm_head(outputs[0]).contiguous()

        ...

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
```

I.e. note all of the tuple concatenations and indexing. The short-term solution for this is to refactor the model to have a fixed architecture (so that you don't need to concatenate and index depending on configuration choices).
The long-term fix is to support `torch.prim.TupleConstruct` in `torch-maximize-value-semantics`.
