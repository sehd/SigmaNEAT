import numba.cuda as cu
from numba.cuda.random import xoroshiro128p_uniform_float32


def system_useGpu():
    return True


def system_maxGenerationCount():
    return 1000


def cudaMethod(isDevice: bool = True):
    def handleCudaMethod(method):
        if(system_useGpu()):
            return cu.jit(func_or_sig=method, device=isDevice)
        else:
            return method
    return handleCudaMethod


@cudaMethod()
def substrate_dimension():
    return 2


@cudaMethod()
def substrate_inputSize():
    return 3


@cudaMethod()
def substrate_outputSize():
    return 1


@cudaMethod()
def substrate_layersCount():
    return 5


@cudaMethod()
def substrate_layerSize():
    return 5


@cudaMethod()
def params_populationSize():
    return 1000


@cudaMethod()
def params_weightThreshold():
    return 0.05


@cudaMethod()
def getSubstrate():
    res = [[(0, j) for j in range(substrate_inputSize())]]
    for i in range(substrate_layersCount()):
        res.append(
            [(i+1, j) for j in range(substrate_layerSize())])
    res.append([(substrate_layersCount()+1, j)
                for j in range(substrate_outputSize())])
    return res


@cudaMethod()
def getInnovationNumber(state):
    thread_id = cu.grid(1)
    return xoroshiro128p_uniform_float32(state, thread_id)
