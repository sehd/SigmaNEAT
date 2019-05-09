import HyperNEAT as NEAT
import numpy as np
from minesweeper import *
from string import ascii_lowercase

class Evaluator(object):
    _gridsize = 0
    _numberofmines = 0
    _maxscore = 0.0
    _bestScore = None

    def __init__(self, gridsize:int, numberofmines:int):
        self._gridsize = gridsize
        self._numberofmines = numberofmines
        self._maxscore = 2 * (gridsize ** 2)
    
    def evaluate(self, genome:NEAT.Genome):
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)
        ms = MineSweeper()
        ms.new_game(self._gridsize,self._numberofmines)
        fitness = self._runboard(net,ms)
        if(self._bestScore is None or fitness > self._bestScore):
            self._bestScore = fitness
            print(f"Best score so far is {self._bestScore}\n")
            ms._showgrid(ms._currentGrid)
        return fitness + 1.0

    def _runboard(self, net:NEAT.NeuralNetwork,ms:MineSweeper):
        score = 0
        while ms.state != GameState.ENDED:
            input = self._get_input(ms)
            net.Input(input)
            net.Activate()
            output = net.Output()
            move = self._get_move(output)
            try:
                score += ms.game_step(move)
            except:
                score -= 1
                if score < -self._maxscore:
                    return -1

        if ms.result == GameResult.WIN:
            return 1
        else:
            return score / self._maxscore

    def _get_input(self, minesweeper):
        res = [1]
        for row in minesweeper._currentGrid:
            res += [self._get_input_item(x) for x in row] 

        return res

    def _get_input_item(self, value):
        if value == ' ' or value == '':
            return -0.8
        elif value == 'F':
            return 0.9
        else:
            return float(value) / 10

    def _get_move(self, output):
        arr = np.array(output)
        action_index = np.argmax(arr)
        flag_index = np.argmin(arr)

        (x,y) = divmod(action_index,self._gridsize)

        return ascii_lowercase[x] + str(y + 1) # + ('F' if output[2] > 0.8 else '')