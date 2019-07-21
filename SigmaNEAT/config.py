import numba.cuda as cu


class Config:

    substrate: dict
    params: dict
    system: dict

    @staticmethod
    def create():
        Config.substrate = {
            "dimension": 2,
            "inputSize": 3,
            "outputSize": 1,
            "layersCount": 5,
            "layerSize": 5,
        }
        Config.substrate["nodes"] = Config.generateSubstrate()
        Config.params = {
            "populationSize": 1000,
            "maxGenerationCount": 1000,
            "weightThreshold": 0.05
        }
        Config.system = {
            "useGpu": True,
        }

    @staticmethod
    def generateSubstrate():
        res = [[(0, j) for j in range(Config.substrate["inputSize"])]]
        for i in range(Config.substrate["layersCount"]):
            res.append(
                [(i+1, j) for j in range(Config.substrate["layerSize"])])
        res.append([(Config.substrate["layersCount"]+1, j)
                    for j in range(Config.substrate["outputSize"])])
        return res


Config.create()


def cudaMethod(isDevice: bool = True):
    def handleCudaMethod(method):
        if(Config.system["useGpu"]):
            return cu.jit(method, isDevice)
        else:
            return method
    return handleCudaMethod
