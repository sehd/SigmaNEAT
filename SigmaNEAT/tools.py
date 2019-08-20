from config import system_useGpu, cudaMethod
import numba.cuda as cu
import numpy as np


@cudaMethod()
def allocateLocalArray(shape: int):
    if(system_useGpu()):
        return cu.local.array(shape, np.float)
    else:
        return np.zeros(shape, np.float)
