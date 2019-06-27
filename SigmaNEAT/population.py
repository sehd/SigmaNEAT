import numba.cuda as cu


class Population(object):
    '''
    A population is responsible to handle and control individuals.
    This includes mutation and cross-over and other GA operations.
    '''
    individuals = []

    def __init__():
        pass

    @cu.jit(device=True)
    def _runGeneration():
        pass
