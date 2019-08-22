from config import System
import numba.cuda as cu


def cudaMethod(isDevice: bool = True):
    def handleCudaMethod(method):
        if(System.useGpu):
            return cu.jit(func_or_sig=method, device=isDevice)
        else:
            return method
    return handleCudaMethod
