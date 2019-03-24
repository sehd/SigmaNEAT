class Evaluator(object):
    
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def Evaluate(self,genome):
        # this creates a neural network (phenotype) from the genome
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)

        # let's input just one pattern to the net, activate it once and get the
        # output
        net.Input([1.0, 0.0, 1.0])
        net.Activate()
        output = net.Output() 

        # the output can be used as any other Python iterable.  For the purposes of
        # the tutorial,
        # we will consider the fitness of the individual to be the neural network
        # that outputs constantly
        # 0.0 from the first output (the second output is ignored)
        fitness = 1.0 - output[0]
        return fitness

