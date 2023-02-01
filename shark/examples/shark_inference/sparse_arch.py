import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.modules.crossnet import LowRankCrossNet
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Dict, List, Optional, Tuple
from torchrec.models.dlrm import (
    choose,
    DenseArch,
    DLRM,
    InteractionArch,
    SparseArch,
    OverArch,
)
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
import numpy as np

torch.manual_seed(0)

np.random.seed(0)


def calculate_offsets(tensor_list, prev_values, prev_offsets):
    offset_init = 0
    offset_list = []
    values_list = []

    if prev_offsets != None:
        offset_init = prev_values.shape[-1]
    for tensor in tensor_list:
        offset_list.append(offset_init)
        offset_init += tensor.shape[0]

    concatendated_tensor_list = torch.cat(tensor_list)

    if prev_values != None:
        concatendated_tensor_list = torch.cat(
            [prev_values, concatendated_tensor_list]
        )

    concatenated_offsets = torch.tensor(offset_list)

    if prev_offsets != None:
        concatenated_offsets = torch.cat([prev_offsets, concatenated_offsets])

    return concatendated_tensor_list, concatenated_offsets


# Have to make combined_keys as dict as to which embedding bags they
# point to. {f1: 0, f3: 0, f2: 1}
# The result will be a triple containing values, indices and pointer tensor.
def to_list(key_jagged, combined_keys):
    key_jagged_dict = key_jagged.to_dict()
    combined_list = []

    for key in combined_keys:
        prev_values, prev_offsets = calculate_offsets(
            key_jagged_dict[key].to_dense(), None, None
        )
        print(prev_values)
        print(prev_offsets)
        combined_list.append(prev_values)
        combined_list.append(prev_offsets)
        combined_list.append(torch.tensor(combined_keys[key]))

    return combined_list


class SparseArchShark(nn.Module):
    def create_emb(self, embedding_dim, num_embeddings_list):
        embedding_list = nn.ModuleList()
        for i in range(0, num_embeddings_list.size):
            num_embeddings = num_embeddings_list[i]
            EE = nn.EmbeddingBag(num_embeddings, embedding_dim, mode="sum")
            W = np.random.uniform(
                low=-np.sqrt(1 / num_embeddings),
                high=np.sqrt(1 / num_embeddings),
                size=(num_embeddings, embedding_dim),
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            embedding_list.append(EE)
        return embedding_list

    def __init__(
        self,
        embedding_dim,
        total_features,
        num_embeddings_list,
    ):
        super(SparseArchShark, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_features = total_features
        self.embedding_list = self.create_emb(
            embedding_dim, num_embeddings_list
        )

    def forward(self, *batched_inputs):
        concatenated_list = []
        input_enum, embedding_enum = 0, 0

        for k in range(len(batched_inputs) // 3):
            values = batched_inputs[input_enum]
            input_enum += 1
            offsets = batched_inputs[input_enum]
            input_enum += 1
            embedding_pointer = int(batched_inputs[input_enum])
            input_enum += 1

            E = self.embedding_list[embedding_pointer]
            V = E(values, offsets)
            concatenated_list.append(V)

        return torch.cat(concatenated_list, dim=1).reshape(
            -1, self.num_features, self.embedding_dim
        )


def test_sparse_arch() -> None:
    D = 3
    eb1_config = EmbeddingBagConfig(
        name="t1",
        embedding_dim=D,
        num_embeddings=10,
        feature_names=["f1", "f3"],
    )
    eb2_config = EmbeddingBagConfig(
        name="t2",
        embedding_dim=D,
        num_embeddings=10,
        feature_names=["f2"],
    )

    ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

    w1 = ebc.embedding_bags["t1"].weight
    w2 = ebc.embedding_bags["t2"].weight

    sparse_arch = SparseArch(ebc)

    keys = ["f1", "f2", "f3", "f4", "f5"]
    offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
    features = KeyedJaggedTensor.from_offsets_sync(
        keys=keys,
        values=torch.tensor(
            [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
        ),
        offsets=offsets,
    )
    sparse_archi = SparseArchShark(D, 3, np.array([10, 10]))
    sparse_archi.embedding_list[0].weight = w1
    sparse_archi.embedding_list[1].weight = w2
    inputs = to_list(features, {"f1": 0, "f3": 0, "f2": 1})

    test_results = sparse_archi(*inputs)
    sparse_features = sparse_arch(features)

    torch.allclose(
        sparse_features,
        test_results,
        rtol=1e-4,
        atol=1e-4,
    )


test_sparse_arch()


class DLRMShark(nn.Module):
    def __init__(
        self,
        embedding_dim,
        total_features,
        num_embeddings_list,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ) -> None:
        super().__init__()

        self.sparse_arch: SparseArchShark = SparseArchShark(
            embedding_dim, total_features, num_embeddings_list
        )
        num_sparse_features: int = total_features

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
        )

        self.inter_arch = InteractionArch(
            num_sparse_features=num_sparse_features,
        )

        over_in_features: int = (
            embedding_dim
            + choose(num_sparse_features, 2)
            + num_sparse_features
        )

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
        )

    def forward(
        self, dense_features: torch.Tensor, *sparse_features
    ) -> torch.Tensor:
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(*sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits


def test_dlrm() -> None:
    B = 2
    D = 8
    dense_in_features = 100

    eb1_config = EmbeddingBagConfig(
        name="t1",
        embedding_dim=D,
        num_embeddings=100,
        feature_names=["f1", "f3"],
    )
    eb2_config = EmbeddingBagConfig(
        name="t2",
        embedding_dim=D,
        num_embeddings=100,
        feature_names=["f2"],
    )

    ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

    sparse_features = KeyedJaggedTensor.from_offsets_sync(
        keys=["f1", "f3", "f2"],
        values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
        offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
    )
    ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
    sparse_nn = DLRM(
        embedding_bag_collection=ebc,
        dense_in_features=dense_in_features,
        dense_arch_layer_sizes=[20, D],
        over_arch_layer_sizes=[5, 1],
    )
    sparse_nn_nod = DLRMShark(
        embedding_dim=8,
        total_features=3,
        num_embeddings_list=np.array([100, 100]),
        dense_in_features=dense_in_features,
        dense_arch_layer_sizes=[20, D],
        over_arch_layer_sizes=[5, 1],
    )

    dense_features = torch.rand((B, dense_in_features))

    x = to_list(sparse_features, {"f1": 0, "f3": 0, "f2": 1})

    w1 = ebc.embedding_bags["t1"].weight
    w2 = ebc.embedding_bags["t2"].weight

    sparse_nn_nod.sparse_arch.embedding_list[0].weight = w1
    sparse_nn_nod.sparse_arch.embedding_list[1].weight = w2

    sparse_nn_nod.dense_arch.load_state_dict(sparse_nn.dense_arch.state_dict())
    sparse_nn_nod.inter_arch.load_state_dict(sparse_nn.inter_arch.state_dict())
    sparse_nn_nod.over_arch.load_state_dict(sparse_nn.over_arch.state_dict())

    logits = sparse_nn(
        dense_features=dense_features,
        sparse_features=sparse_features,
    )
    logits_nod = sparse_nn_nod(dense_features, *x)

    # print(logits)
    # print(logits_nod)

    # Import the module and print.
    mlir_importer = SharkImporter(
        sparse_nn_nod,
        (dense_features, *x),
        frontend="torch",
    )

    (dlrm_mlir, func_name), inputs, golden_out = mlir_importer.import_debug(
        tracing_required=True
    )

    shark_module = SharkInference(
        dlrm_mlir, func_name, device="cpu", mlir_dialect="linalg"
    )
    shark_module.compile()
    result = shark_module.forward(inputs)
    np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)

    torch.allclose(
        logits,
        logits_nod,
        rtol=1e-4,
        atol=1e-4,
    )


test_dlrm()
