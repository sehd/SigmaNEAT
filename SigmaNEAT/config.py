import numba.cuda as cu


def cudaMethod(isDevice: bool = True):
    def handleCudaMethod(method):
        if(System.useGpu):
            return cu.jit(func_or_sig=method, device=isDevice)
        else:
            return method
    return handleCudaMethod


class System:
    useGpu = True
    maxGenerationCount = 1000
    threadsPerBlock = 32
    blocksPerGrid = (maxGenerationCount +
                     (threadsPerBlock - 1)) // threadsPerBlock


class Substrate:
    dimension = 2
    inputSize = 3
    outputSize = 1
    layersCount = 5
    layerSize = 5


class Params:
    populationSize = 1000
    weightThreshold = 0.05
