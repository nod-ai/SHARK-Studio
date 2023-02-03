# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019


import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter


torch.manual_seed(0)
np.random.seed(0)


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill

            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)

            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            EE = nn.EmbeddingBag(n, m, mode="sum")
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            print(W)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
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
        weighted_pooling=None,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments
            self.output_d = 0
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling

            # create operators
            self.emb_l, w_list = self.create_emb(
                m_spa, ln_emb, weighted_pooling
            )
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList()
                for w in w_list:
                    self.v_W_l.append(nn.Parameter(w))
            else:
                self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
        # TORCH-MLIR
        # We are passing all the embeddings as arguments for easy parsing.

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(
                    0, sparse_index_group_batch
                )
            else:
                per_sample_weights = None

            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=per_sample_weights,
            )

            ly.append(V)

        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor(
                [i for i in range(ni) for j in range(i + offset)]
            )
            lj = torch.tensor(
                [j for i in range(nj) for j in range(i + offset)]
            )
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, *lS_i):
        return self.sequential_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # # clamp output if needed
        # if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
        # z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        # else:
        # z = p

        return p


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


# model related parameters
parser = argparse.ArgumentParser(
    description="Train Deep Learning Recommendation Model (DLRM)"
)
parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
parser.add_argument(
    "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
)
# j will be replaced with the table number
parser.add_argument(
    "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2"
)
parser.add_argument(
    "--arch-mlp-top", type=dash_separated_ints, default="8-2-1"
)
parser.add_argument(
    "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
)
parser.add_argument(
    "--arch-interaction-itself", action="store_true", default=False
)
parser.add_argument("--weighted-pooling", type=str, default=None)

args = parser.parse_args()

ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
ln_top = np.fromstring(args.arch_mlp_top, dtype=int, sep="-")
m_den = ln_bot[0]
ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
m_spa = args.arch_sparse_feature_size
ln_emb = np.asarray(ln_emb)
num_fea = ln_emb.size + 1  # num sparse + num dense features


# Initialize the model.
dlrm_model = DLRM_Net(
    m_spa=m_spa,
    ln_emb=ln_emb,
    ln_bot=ln_bot,
    ln_top=ln_top,
    arch_interaction_op=args.arch_interaction_op,
)


# Inputs to the model.
dense_inp = torch.tensor([[0.6965, 0.2861, 0.2269, 0.5513]])
vs0 = torch.tensor([[0], [0], [0]], dtype=torch.int64)
vsi = torch.tensor([1, 2, 3]), torch.tensor([1]), torch.tensor([1])

input_dlrm = (dense_inp, vs0, *vsi)

golden_output = dlrm_model(dense_inp, vs0, *vsi)

mlir_importer = SharkImporter(
    dlrm_model,
    input_dlrm,
    frontend="torch",
)

(dlrm_mlir, func_name), inputs, golden_out = mlir_importer.import_debug(
    tracing_required=True
)

shark_module = SharkInference(
    dlrm_mlir, func_name, device="vulkan", mlir_dialect="linalg"
)
shark_module.compile()
result = shark_module.forward(input_dlrm)
np.testing.assert_allclose(
    golden_output.detach().numpy(), result, rtol=1e-02, atol=1e-03
)


# Verified via torch-mlir.
# import torch_mlir
# from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


# module = torch_mlir.compile(
# dlrm_model, inputs, use_tracing=True, output_type="linalg-on-tensors"
# )
# backend = refbackend.RefBackendLinalgOnTensorsBackend()
# compiled = backend.compile(module)
# jit_module = backend.load(compiled)

# dense_numpy = dense_inp.numpy()
# vs0_numpy = vs0.numpy()
# vsi_numpy = [inp.numpy() for inp in vsi]

# numpy_inp = (dense_numpy, vs0_numpy, *vsi_numpy)

# print(jit_module.forward(*numpy_inp))
