import numba.cuda as cu


def cudaMethod():
    def handleCudaMethod(method):
        if(SYSTEM__USE_GPU):
            return cu.jit(func_or_sig=method, device=True)
        else:
            return method
    return handleCudaMethod


# System
SYSTEM__USE_GPU = True
SYSTEM__MAX_GENERATION_COUNT = 1000
SYSTEM__THREADS_PER_BLOCK = 512

# Substrate
SUBSTRATE__DIMENSION = 2
SUBSTRATE__INPUT_SIZE = 3
SUBSTRATE__OUTPUT_SIZE = 1
SUBSTRATE__LAYERS_COUNT = 5
SUBSTRATE__LAYER_SIZE = 5


# Params
PARAMS__POPULATION_SIZE = 1000
PARAMS__WEIGHT_THRESHOLD = 0.05
