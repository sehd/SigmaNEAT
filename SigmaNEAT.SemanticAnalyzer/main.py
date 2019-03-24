import HyperNEAT as NEAT

params = NEAT.Parameters()  
# params.PopulationSize = 100

genome = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.TANH, 0, params) 
pop = NEAT.Population(genome, params, True, 1.0, 0) # the 0 is the RNG seed

for generation in range(100): # run for 100 generations
    # retrieve a list of all genomes in the population
    genome_list = []
    for s in pop.Species:
        genome_list+=s.Individuals

    # apply the evaluation function to all genomes
    for genome in genome_list:
        fitness = evaluate(genome)
        genome.SetFitness(fitness)

    print("Best Fitness: " + str(pop.GetBestFitnessEver()))
    print("Best Genome: " + str(pop.GetBestGenome()))

    # advance to the next generation
    pop.Epoch()