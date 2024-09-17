#MIT License
#
#Copyright (c) Facebook, Inc. and its affiliates.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch_mlir
from torch.utils.data import Dataset
from numpy import random as ra

from shark.shark_importer import SharkImporter
from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from tank.model_utils import compare_tensors
from shark.parser import shark_args

import unittest
import pytest

import torch.nn.functional as F
from torch.nn.parameter import Parameter

class QREmbeddingBag(nn.Module):
    r"""Computes sums or means over two 'bags' of embeddings, one using the quotient
    of the indices and the other using the remainder of the indices, without
    instantiating the intermediate embeddings, then performs an operation to combine these.

    For bags of constant length and no :attr:`per_sample_weights`, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=0)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=0)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=0)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    QREmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights`` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Known Issues:
    Autograd breaks with multiple GPUs. It breaks only with multiple embeddings.

    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table. If ``"add"``
                              or ``"mult"`` operation are used, these embedding dimensions must be
                              the same. If a single embedding_dim is used, then it will use this
                              embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    Attributes:
        weight (Tensor): the learnable weights of each embedding table is the module of shape
                         `(num_embeddings, embedding_dim)` initialized using a uniform distribution
                         with sqrt(1 / num_categories).

    Inputs: :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
        :attr:`per_index_weights` (Tensor, optional)

        - If :attr:`input` is 2D of shape `(B, N)`,

          it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
          this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
          :attr:`offsets` is ignored and required to be ``None`` in this case.

        - If :attr:`input` is 1D of shape `(N)`,

          it will be treated as a concatenation of multiple bags (sequences).
          :attr:`offsets` is required to be a 1D tensor containing the
          starting index positions of each bag in :attr:`input`. Therefore,
          for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
          having ``B`` bags. Empty bags (i.e., having 0-length) will have
          returned vectors filled by zeros.

        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.


    Output shape: `(B, embedding_dim)`

    """
    __constants__ = ['num_categories', 'embedding_dim', 'num_collisions',
                     'operation', 'max_norm', 'norm_type', 'scale_grad_by_freq',
                     'mode', 'sparse']

    def __init__(self, num_categories, embedding_dim, num_collisions,
                 operation='mult', max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, mode='mean', sparse=False,
                 _weight=None):
        super(QREmbeddingBag, self).__init__()

        assert operation in ['concat', 'mult', 'add'], 'Not valid operation!'

        self.num_categories = num_categories
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions
        self.operation = operation
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if self.operation == 'add' or self.operation == 'mult':
            assert self.embedding_dim[0] == self.embedding_dim[1], \
                'Embedding dimensions do not match!'

        self.num_embeddings = [int(np.ceil(num_categories / num_collisions)),
            num_collisions]

        if _weight is None:
            self.weight_q = Parameter(torch.Tensor(self.num_embeddings[0], self.embedding_dim[0]))
            self.weight_r = Parameter(torch.Tensor(self.num_embeddings[1], self.embedding_dim[1]))
            self.reset_parameters()
        else:
            assert list(_weight[0].shape) == [self.num_embeddings[0], self.embedding_dim[0]], \
                'Shape of weight for quotient table does not match num_embeddings and embedding_dim'
            assert list(_weight[1].shape) == [self.num_embeddings[1], self.embedding_dim[1]], \
                'Shape of weight for remainder table does not match num_embeddings and embedding_dim'
            self.weight_q = Parameter(_weight[0])
            self.weight_r = Parameter(_weight[1])
        self.mode = mode
        self.sparse = sparse

    def reset_parameters(self):
        nn.init.uniform_(self.weight_q, np.sqrt(1 / self.num_categories))
        nn.init.uniform_(self.weight_r, np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None, per_sample_weights=None):
        input_q = (input / self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()

        embed_q = F.embedding_bag(input_q, self.weight_q, offsets, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq, self.mode,
                                  self.sparse, per_sample_weights)
        embed_r = F.embedding_bag(input_r, self.weight_r, offsets, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq, self.mode,
                                  self.sparse, per_sample_weights)

        if self.operation == 'concat':
            embed = torch.cat((embed_q, embed_r), dim=1)
        elif self.operation == 'add':
            embed = embed_q + embed_r
        elif self.operation == 'mult':
            embed = embed_q * embed_r

        return embed

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        return s.format(**self.__dict__)


def md_solver(n, alpha, d0=None, B=None, round_dim=True, k=None):
    '''
    An external facing function call for mixed-dimension assignment
    with the alpha power temperature heuristic
    Inputs:
    n -- (torch.LongTensor) ; Vector of num of rows for each embedding matrix
    alpha -- (torch.FloatTensor); Scalar, non-negative, controls dim. skew
    d0 -- (torch.FloatTensor); Scalar, baseline embedding dimension
    B -- (torch.FloatTensor); Scalar, parameter budget for embedding layer
    round_dim -- (bool); flag for rounding dims to nearest pow of 2
    k -- (torch.LongTensor) ; Vector of average number of queries per inference
    '''
    n, indices = torch.sort(n)
    k = k[indices] if k is not None else torch.ones(len(n))
    d = alpha_power_rule(n.type(torch.float) / k, alpha, d0=d0, B=B)
    if round_dim:
        d = pow_2_round(d)
    undo_sort = [0] * len(indices)
    for i, v in enumerate(indices):
        undo_sort[v] = i
    return d[undo_sort]


def alpha_power_rule(n, alpha, d0=None, B=None):
    if d0 is not None:
        lamb = d0 * (n[0].type(torch.float) ** alpha)
    elif B is not None:
        lamb = B / torch.sum(n.type(torch.float) ** (1 - alpha))
    else:
        raise ValueError("Must specify either d0 or B")
    d = torch.ones(len(n)) * lamb * (n.type(torch.float) ** (-alpha))
    for i in range(len(d)):
        if i == 0 and d0 is not None:
            d[i] = d0
        else:
            d[i] = 1 if d[i] < 1 else d[i]
    return (torch.round(d).type(torch.long))


def pow_2_round(dims):
    return 2 ** torch.round(torch.log2(dims.type(torch.float)))


class PrEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, base_dim):
        super(PrEmbeddingBag, self).__init__()
        self.embs = nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=False)
        torch.nn.init.xavier_uniform_(self.embs.weight)
        if embedding_dim < base_dim:
            self.proj = nn.Linear(embedding_dim, base_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif embedding_dim == base_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError(
                "Embedding dim " + str(embedding_dim) + " > base dim " + str(base_dim)
            )

    def forward(self, input, offsets=None, per_sample_weights=None):
        return self.proj(self.embs(
            input, offsets=offsets, per_sample_weights=per_sample_weights))


def unpack_batch(b):
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

class DLRM_Net(nn.Module):
    
    emb_l_q: List[torch.Tensor]
    def create_mlp(self, ln, sigmoid_layer):

        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            mean = 0.0  
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)

            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):

            n = ln[i]
            n = n.item()

            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=False,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False)
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)

                EE.weight.data = torch.tensor(W, requires_grad=True)

            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce"
    ):
        super(DLRM_Net, self).__init__()

        self.emb_l_q: List[torch.Tensor] = []
        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function=loss_function
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # create operators
            if ndevices <= 1:
                print("ndev")
                print(ln_emb)
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q: List[torch.Tensor] = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i):

        ly = []
        for k, E in enumerate(self.emb_l):
            sparse_index_group_batch = lS_i[k]
            sparse_offset_group_batch = lS_o[k]

            if self.v_W_l[k] is not None:
                per_sample_weights = self.v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].numel()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].numel()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                    ly.append(QV)
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                    ly.append(QV)
            else:
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )

                ly.append(V)

        return ly

    def quantize_embedding(self, bits):

        n = len(self.emb_l)
        self.emb_l_q: List[torch.Tensor] = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x: torch.Tensor, ly: List[torch.Tensor]):

        if self.arch_interaction_op == "dot":
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            lli: List[int] = []
            llj: List[int] = []
            for idx in range(ni):
                for j in range(idx + offset):
                    lli.append(idx)
            li = torch.tensor(lli)

            for idx in range(nj):
                for j in range(idx + offset):
                    llj.append(j)
            lj = torch.tensor(llj)

            Zflat = Z[:, li, lj]

            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            raise Exception(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def new_forward(self, dense_x, lS_o, lS_i):
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)

    def forward(self, *args):
        dense_x = args[0]
        lS_o = args[1]
        lS_i = []
        for i in range(2, len(args)):
            lS_i.append(args[i])

        return self.new_forward(dense_x, lS_o, lS_i)

    def apply_mlp_bot(self, x):
        return self.bot_l(x)

    def apply_mlp_top(self, x):
        return self.top_l(x)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp_bot(dense_x)

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i)

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp_top(z)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

def collate_wrapper_random_offset(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)

def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)


# random data from uniform or gaussian ditribution (input data)
def generate_dist_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    rand_data_dist,
    rand_data_min,
    rand_data_max,
    rand_data_mu,
    rand_data_sigma,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            if rand_data_dist == "gaussian":
                if rand_data_mu == -1:
                    rand_data_mu = (rand_data_max + rand_data_min) / 2.0
                r = ra.normal(rand_data_mu, rand_data_sigma, sparse_group_size)
                sparse_group = np.clip(r, rand_data_min, rand_data_max)
                sparse_group = np.unique(sparse_group).astype(np.int64)
            elif rand_data_dist == "uniform":
                r = ra.random(sparse_group_size)
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            else:
                raise(rand_data_dist, "distribution is not supported. \
                     please select uniform or gaussian")

            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)

def make_random_data_and_loader(args, ln_emb, m_den,
    offset_to_length_converter=False,
):

    train_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed
    )  # WARNING: generates a batch of lookups at once

    test_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed
    )

    collate_wrapper_random = collate_wrapper_random_offset
    if offset_to_length_converter:
        collate_wrapper_random = collate_wrapper_random_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_data, train_loader, test_data, test_loader


def generate_random_data(
    m_den,
    ln_emb,
    data_size,
    num_batches,
    mini_batch_size,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    num_targets=1,
    round_targets=False,
    data_generation="random",
    trace_file="",
    enable_padding=False,
    length=False, # length for caffe2 version (except dlrm_s_caffe2)
):
    nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
    if num_batches != 0:
        nbatches = num_batches
        data_size = nbatches * mini_batch_size
    # print("Total number of batches %d" % nbatches)

    # inputs
    lT = []
    lX = []
    lS_offsets = []
    lS_indices = []
    for j in range(0, nbatches):
        # number of data points in a batch
        n = min(mini_batch_size, data_size - (j * mini_batch_size))

        # generate a batch of dense and sparse features
        if data_generation == "random":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_uniform_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                length,
            )
        elif data_generation == "synthetic":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_synthetic_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                trace_file,
                enable_padding
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + data_generation + " is not supported"
            )
        # dense feature
        lX.append(Xt)
        # sparse feature (sparse indices)
        lS_offsets.append(lS_emb_offsets)
        lS_indices.append(lS_emb_indices)

        # generate a batch of target (probability of a click)
        P = generate_random_output_batch(n, num_targets, round_targets)
        lT.append(P)

    return (nbatches, lX, lS_offsets, lS_indices, lT)

# uniform ditribution (input data)
class RandomDataset(Dataset):

    def __init__(
            self,
            m_den,
            ln_emb,
            data_size,
            num_batches,
            mini_batch_size,
            num_indices_per_lookup,
            num_indices_per_lookup_fixed,
            num_targets=1,
            round_targets=False,
            data_generation="random",
            trace_file="",
            enable_padding=False,
            reset_seed_on_access=False,
            rand_data_dist="uniform",
            rand_data_min=1,
            rand_data_max=1,
            rand_data_mu=-1,
            rand_data_sigma=1,
            rand_seed=0
    ):
        # compute batch size
        nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
        if num_batches != 0:
            nbatches = num_batches
            data_size = nbatches * mini_batch_size
            # print("Total number of batches %d" % nbatches)

        # save args (recompute data_size if needed)
        self.m_den = m_den
        self.ln_emb = ln_emb
        self.data_size = data_size
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.num_indices_per_lookup_fixed = num_indices_per_lookup_fixed
        self.num_targets = num_targets
        self.round_targets = round_targets
        self.data_generation = data_generation
        self.trace_file = trace_file
        self.enable_padding = enable_padding
        self.reset_seed_on_access = reset_seed_on_access
        self.rand_seed = rand_seed
        self.rand_data_dist = rand_data_dist
        self.rand_data_min = rand_data_min
        self.rand_data_max = rand_data_max
        self.rand_data_mu = rand_data_mu
        self.rand_data_sigma = rand_data_sigma

    def reset_numpy_seed(self, numpy_rand_seed):
        np.random.seed(numpy_rand_seed)
        # torch.manual_seed(numpy_rand_seed)

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # WARNING: reset seed on access to first element
        # (e.g. if same random samples needed across epochs)
        if self.reset_seed_on_access and index == 0:
            self.reset_numpy_seed(self.rand_seed)

        # number of data points in a batch
        n = min(self.mini_batch_size, self.data_size - (index * self.mini_batch_size))

        # generate a batch of dense and sparse features
        if self.data_generation == "random":
            (X, lS_o, lS_i) = generate_dist_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                rand_data_dist=self.rand_data_dist,
                rand_data_min=self.rand_data_min,
                rand_data_max=self.rand_data_max,
                rand_data_mu=self.rand_data_mu,
                rand_data_sigma=self.rand_data_sigma,
            )
        elif self.data_generation == "synthetic":
            (X, lS_o, lS_i) = generate_synthetic_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                self.trace_file,
                self.enable_padding
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + self.data_generation + " is not supported"
            )

        # generate a batch of target (probability of a click)
        T = generate_random_output_batch(n, self.num_targets, self.round_targets)

        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


class args:
    def __init__(self):
        self.arch_sparse_feature_size     = 2
        self.arch_embedding_size          = "4-3-2"
        self.arch_mlp_bot                 = "4-3-2"
        self.arch_mlp_top                 = "4-2-1"
        self.arch_interaction_op          = "dot"
        self.arch_interaction_itself      = False
        self.weighted_pooling             = None
        self.md_flag                      = False
        self.md_threshold                 = 200
        self.md_temperature               = 0.3
        self.md_round_dims                = False
        self.qr_flag                      = False
        self.qr_threshold                 = 200
        self.qr_operation                 = "mult"
        self.qr_collisions                = 4
        self.activation_function          = "relu"
        self.loss_function                = "mse"
        self.loss_weights                 = "1.0-1.0"
        self.loss_threshold               = 0.0
        self.round_targets                = False
        self.data_size                    = 1
        self.num_batches                  = 0 
        self.data_generation              = "random" 
        self.rand_data_dist               = "uniform"
        self.rand_data_min                = 0
        self.rand_data_max                = 1
        self.rand_data_mu                 = -1
        self.rand_data_sigma              = 1
        self.data_trace_file              = "./input/dist_emb_j.log"
        self.data_set                     = "kaggle"
        self.raw_data_file                = ""
        self.processed_data_file          = ""
        self.data_randomize               = "total"
        self.data_trace_enable_padding    = False
        self.max_ind_range                = -1
        self.data_sub_sample_rate         = 0.0
        self.num_indices_per_lookup       = 10
        self.num_indices_per_lookup_fixed = False
        self.num_workers                  = 0
        self.memory_map                   = False
        self.mini_batch_size              = 1 
        self.nepochs                      = 1
        self.learning_rate                = 0.01
        self.print_precision              = 5
        self.numpy_rand_seed              = 123
        self.sync_dense_params            = True
        self.optimizer                    = "sgd"
        self.dataset_multiprocessing      = False
        self.inference_only               = False
        self.quantize_mlp_with_bit        = 32
        self.quantize_emb_with_bit        = 32
        self.save_onnx                    = False
        self.use_gpu                      = False
        self.local_rank                   = -1
        self.dist_backend                 = ""
        self.print_freq                   = 1
        self.test_freq                    = -1
        self.test_mini_batch_size         = -1
        self.test_num_workers             = -1
        self.print_time                   = False
        self.print_wall_time              = False 
        self.debug_mode                   = False
        self.enable_profiling             = False
        self.plot_compute_graph           = False
        self.tensor_board_filename        = "run_kaggle_pt"
        self.save_model                   = ""
        self.load_model                   = ""
        self.mlperf_logging               = False
        self.mlperf_acc_threshold         = 0.0
        self.mlperf_auc_threshold         = 0.0
        self.mlperf_bin_loader            = False
        self.mlperf_bin_shuffle           = False
        self.mlperf_grad_accum_iter       = 1
        self.lr_num_warmup_steps          = 0
        self.lr_decay_start_step          = 0
        self.lr_num_decay_steps           = 0
        self.ln_emb                       = []

def generate_dlrm_model_and_inputs():

    global args
    global nbatches
    global nbatches_test
    global writer
    args = args()

    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)
    #device  = torch.device("cpu")
    use_gpu = args.use_gpu

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")

    #Random Data
    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    print(ln_emb)
    m_den = ln_bot[0]
    train_data, train_ld, test_data, test_ld = make_random_data_and_loader(args, ln_emb, m_den)
    nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
    nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist()

    print(args.ln_emb)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()

    ln_emb = np.asarray(ln_emb)
    print(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    global ndevices
    ndevices = -1
    global dlrm
    print(ln_emb)
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function
    )
    
    #get all the batches, but just use one for getting the trace
    batches = [batch for batch in train_ld]
    
    #we can just process one batch to get the Torcscript, no need to go over all the batches.
    X, lS_o, lS_i, T, W, CBPP = unpack_batch(batches[0])

    #prepare the trace inputs
    model_inputs = []
    model_inputs.append(X.to(torch.float))
    model_inputs.append(lS_o)

    for i in range(len(lS_i)):
        model_inputs.append(lS_i[i])

    actual_out = dlrm(*model_inputs)

    return dlrm, model_inputs, actual_out

####################################################### Dlrm Module Tester ####################################################

class DlrmModuleTester:
    def __init__(
        self,
        dynamic=False,
        device="cpu",
        save_mlir=False,
        save_vmfb=False,
    ):
        self.dynamic = dynamic
        self.device = device
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self):
        model, input, act_out = generate_dlrm_model_and_inputs()
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        mlir_importer = SharkImporter(
            model,
            (input),
            frontend="torch",
        )
        dlrm_mlir, func_name = mlir_importer.import_mlir(tracing_required=True)

        shark_module = SharkInference(
            dlrm_mlir, func_name, device=self.device, mlir_dialect="linalg"
        )

        shark_module.compile()
        results = shark_module.forward((input))
        assert True == compare_tensors(act_out, results)


class DlrmModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = DlrmModuleTester(self)

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()
        
####################################### QREmbeddingBag Tester ############################################# (quotient remainder Mode)

def generate_qr_embedding_bag():
    
    EE = QREmbeddingBag(
                    10,
                    100,
                    4,
                    "mult",
                    mode="sum",
                    sparse=False,
                )
    
    inputs = torch.tensor([1, 2])
    offset = torch.tensor([0])
    
    actual_out = EE(inputs, offset)
    
    return EE, [inputs, offset], actual_out    


class QREmbeddingBagModuleTester:
    def __init__(
        self,
        dynamic=False,
        device="cpu",
        save_mlir=False,
        save_vmfb=False,
    ):
        self.dynamic = dynamic
        self.device = device
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self):
        model, input, act_out = generate_qr_embedding_bag()
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        mlir_importer = SharkImporter(
            model,
            (input),
            frontend="torch",
        )
        qr_embedding_bag_mlir, func_name = mlir_importer.import_mlir(tracing_required=True)

        shark_module = SharkInference(
            qr_embedding_bag_mlir, func_name, device=self.device, mlir_dialect="linalg"
        )

        shark_module.compile()
        results = shark_module.forward((input))
        assert True == compare_tensors(act_out, results)
        

class QREmbeddingBagModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = QREmbeddingBagModuleTester(self)

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()
        
################################# PREmbeddingBag Tester ################################### (mixed dimensions)
        
def generate_pr_embedding_bag(): 
    
    n = 100
    _m = 10
    base = 10
    
    EE = PrEmbeddingBag(n, _m, base)
    
    W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)).astype(np.float32)
    
    EE.embs.weight.data = torch.tensor(W, requires_grad=True)
    
    inputs = torch.tensor([1, 2])
    offset = torch.tensor([0])
    
    actual_out = EE(inputs, offset)
    
    return EE, [inputs, offset], actual_out

class PREmbeddingBagModuleTester:
    def __init__(
        self,
        dynamic=False,
        device="cpu",
        save_mlir=False,
        save_vmfb=False,
    ):
        self.dynamic = dynamic
        self.device = device
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self):
        model, input, act_out = generate_pr_embedding_bag()
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        mlir_importer = SharkImporter(
            model,
            (input),
            frontend="torch",
        )
        pr_embedding_bag_mlir, func_name = mlir_importer.import_mlir(tracing_required=True)

        shark_module = SharkInference(
            pr_embedding_bag_mlir, func_name, device=self.device, mlir_dialect="linalg"
        )

        shark_module.compile()
        results = shark_module.forward((input))
        assert True == compare_tensors(act_out, results)
        

class PREmbeddingBagModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = PREmbeddingBagModuleTester(self)

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()



if __name__ == "__main__":
    unittest.main()

    #comment unittest.main() and uncomment code below to compile directly via torch_mlir.compile
    #model, inputs, actual_inputs = generate_dlrm_model_and_inputs()
    #script_module = torch.jit.script(model, inputs)
    #linalg_module = torch_mlir.compile(script_module, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    #print(linalg_module)

