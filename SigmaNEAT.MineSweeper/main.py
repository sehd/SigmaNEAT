import HyperNEAT as NEAT
from evaluator import Evaluator
params = NEAT.Parameters()  
# params.PopulationSize = 100
gridsize = 10
minecount = 20

genome = NEAT.Genome(0, gridsize ** 2 + 1, 0, gridsize ** 2, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.TANH, 0, params, 0) 
pop = NEAT.Population(genome, params, True, 1.0, 0)
evaluator = Evaluator(gridsize,minecount)

for generation in range(1000): 
    genome_list = []
    for s in pop.Species:
        for genome in s.Individuals:
            fitness = evaluator.evaluate(genome)
            genome.SetFitness(fitness)

    print("Best Fitness: " + str(pop.GetBestFitnessEver()))
    print("Best Genome: " + str(pop.GetBestGenome()))

    pop.Epoch()


#from minesweeper import MineSweeper

#ms=MineSweeper()
#ms.new_game(10,5)
#ms.play_user_game()