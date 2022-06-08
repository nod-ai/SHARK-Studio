import torch
from shark.shark_inference import SharkInference
import multiprocessing as mp
import cupy as cp
from cupy.cuda import nccl
from cupy import cuda


class MulModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.train(False)

    def forward(self, x):
        return torch.mul(x, x)


def generate_module(dummy_input, device_idx):
    return module


def run_inference(args):
    argmap = {"idx": args[0], "dummy_input": args[1], "results": args[2]}
    #Iree modules aren't currently pickleable, but we can loop through inputs that a read in each subprocess
    module = SharkInference(MulModule(), (dummy_input,),
                            device="gpu",
                            device_idx=argmap["idx"])
    module.compile()
    #Ideally we would instead get a ptr to the cuda tensor that is the result prior to transfer back to cpu
    argmap["results"].put(module.forward((argmap["dummy_input"],)))


num_devices = 2
dummy_input = torch.randn(512)

with mp.Manager() as m:
    results = m.Queue()
    processes = []
    args = [[str(idx), dummy_input, results] for idx in range(num_devices)]
    for i in range(num_devices):
        p = mp.Process(target=run_inference, args=[args[i]])
        p.start()
        processes.append(p)

    [p.join() for p in processes]
    devs = [0, 1]
    comms = nccl.NcclCommunicator.initAll(devs)
    nccl.groupStart()
    for comm in comms:
        dev_id = comm.device_id()
        rank = comm.rank_id()
        assert rank == dev_id
        recvbuf = cp.ndarray(shape=dummy_input.shape, dtype=cp.float32)
        if rank == 0:
            with cuda.Device(dev_id):
                sendbuf = cp.array(results.get())
                comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, 512,
                               nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                               cuda.Stream.null.ptr)
        elif rank == 1:
            with cuda.Device(dev_id):
                sendbuf = cp.array(results.get())
                comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, 512,
                               nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                               cuda.Stream.null.ptr)
    nccl.groupEnd()

golden = MulModule()(dummy_input)
assert (torch.allclose(torch.tensor(cp.asnumpy(recvbuf)), golden * 2))
