import numba.cuda as cu

system = {
    "useGpu": True,
    "maxGenerationCount": 1000,
}


def cudaMethod(isDevice: bool = True):
    def handleCudaMethod(method):
        if(system["useGpu"]):
            return cu.jit(func_or_sig=method, device=isDevice)
        else:
            return method
    return handleCudaMethod


@cudaMethod()
def getConfigs():
    res = {}

    res["substrate_dimension"] = 2,
    res["substrate_inputSize"] = 3,
    res["substrate_outputSize"] = 1,
    res["substrate_layersCount"] = 5,
    res["substrate_layerSize"] = 5,
    res["params_populationSize"] = 1000,
    res["params_weightThreshold"] = 0.05,
    res["params_innovationNumber"] = 0

    res["substrate_nodes"] = generateSubstrate(res["substrate"])
    return res


@cudaMethod()
def generateSubstrate(substrate):
    res = [[(0, j) for j in range(substrate["inputSize"])]]
    for i in range(substrate["layersCount"]):
        res.append(
            [(i+1, j) for j in range(substrate["layerSize"])])
    res.append([(substrate["layersCount"]+1, j)
                for j in range(substrate["outputSize"])])
    return res


def getInnovationNumber():
    getConfigs()["params"]["innovationNumber"] += 1
    return getConfigs()["params"]["innovationNumber"]
