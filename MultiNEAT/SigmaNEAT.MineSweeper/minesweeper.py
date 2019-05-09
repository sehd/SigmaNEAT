import random
import re
import time
from string import ascii_lowercase
from enum import Enum

class GameState(Enum):
    NEW = 1
    PLAYING = 2
    ENDED = 3

class GameResult(Enum):
    WIN = 1
    LOSE = 2

class MineSweeper:
    state = GameState.NEW
    result = None
    _gridsize = 0
    _numberofmines = 0
    _currentGrid = []
    _grid = []
    _flags = []
    _mines = []
    _starttime = 0

    def new_game(self, gridsize:int, numberofmines:int):
        self._gridsize = gridsize
        self._numberofmines = numberofmines
        self._currentGrid = [[' ' for i in range(gridsize)] for i in range(gridsize)]
        self._grid = []
        self._flags = []
        self._mines = []
        self._starttime = 0
        self.state = GameState.NEW
        self.result = None

    def play_user_game(self):
        if(self.state != GameState.NEW):
            print('Game state is invalid')
            return

        helpmessage = ("Type the column followed by the row (eg. a5). "
                       "To put or remove a flag, add 'f' to the cell (eg. a5f).")
        self._showgrid(self._currentGrid)
        print(helpmessage + " Type 'help' to show this message again.\n")
    
        while self.state != GameState.ENDED:
            minesleft = self._numberofmines - len(self._flags)
            prompt = input('Enter the cell ({} mines left): '.format(minesleft))
            if prompt == 'help':
                message = helpmessage
            else:
                try:
                    self.game_step(prompt)
                    print('\n\n')
                    self._showgrid(self._currentGrid)
                except Exception as ex:
                    if str(ex) == 'Invalid Input':
                        print("Invalid cell. " + helpmessage)
                    else:
                        print(str(ex))

                    print('\n\n')
                    self._showgrid(self._currentGrid)

        if(self.result == GameResult.LOSE):
            print('Game Over\n')
            self._showgrid(self._grid)
            if self._playagain():
                self.new_game(self._gridsize, self._numberofmines)
                self.play_user_game()
        else:
            minutes, seconds = divmod(int(time.time() - self._starttime), 60)
            print('You Win.\nIt took you {} minutes and {} seconds.\n'.format(minutes, seconds))
            self._showgrid(self._grid)
            if self._playagain():
                self.new_game(self._gridsize, self._numberofmines)
                self.play_user_game()

    def game_step(self, move:str):
        parse_result = self._parseinput(move)
        
        if not parse_result['isSuccessful']:
            raise Exception('Invalid Input')
        
        currcell_coordinates = parse_result['cell']
        rowno, colno = currcell_coordinates 
    
        currcell = self._currentGrid[rowno][colno]
        flag = parse_result['flag']
    
        if self.state == GameState.NEW:
            self._grid, self._mines = self._setupgrid(currcell_coordinates)
            self._starttime = time.time()
            self.state = GameState.PLAYING
    
        if flag:
            self._add_flag(currcell, rowno, colno)
            if set(self._flags) == set(self._mines):
                self.state = GameState.ENDED
                self.result = GameResult.WIN
            return 0
        elif currcell_coordinates in self._flags:
            raise Exception('There is a flag there')
        elif self._grid[rowno][colno] == 'X':
            self.state = GameState.ENDED
            self.result = GameResult.LOSE
            return 0
        elif currcell == ' ':
            res = self._showcells(rowno, colno)

            win = True
            for i in range(self._gridsize):
                for j in range(self._gridsize):
                    if self._currentGrid[i][j] == ' ' and (i,j) not in self._mines:
                        win = False
                        break
                if not win:
                    break
            if win:
                self.state = GameState.ENDED
                self.result = GameResult.WIN
                
            return res
        else:
            raise Exception("That cell is already shown")

    def _add_flag(self, cell, rowno, colno):
        if cell == ' ':
            self._currentGrid[rowno][colno] = 'F'
            self._flags.append((rowno, colno))
        elif cell == 'F':
            self._currentGrid[rowno][colno] = ' '
            self._flags.remove((rowno, colno))
        else:
            raise Exception('Cannot put a flag there')

    def _setupgrid(self, startCell):
        emptygrid = [['0' for i in range(self._gridsize)] for i in range(self._gridsize)]
        mines = self._getmines(emptygrid, startCell)
    
        for i, j in mines:
            emptygrid[i][j] = 'X'
    
        grid = self._getnumbers(emptygrid)
        return (grid, mines)
    
    def _showgrid(self, grid):
        horizontal = '   ' + (4 * self._gridsize * '-') + '-'
        toplabel = '     '
    
        for i in ascii_lowercase[:self._gridsize]:
            toplabel = toplabel + i + '   '
    
        print(toplabel + '\n' + horizontal)
    
        for idx, i in enumerate(grid):
            row = '{0:2} |'.format(idx + 1)
            for j in i:
                row = row + ' ' + j + ' |'
    
            print(row + '\n' + horizontal)

        print('')
    
    def _getrandomcell(self, grid):
        a = random.randint(0, self._gridsize - 1)
        b = random.randint(0, self._gridsize - 1)
        return (a, b)
    
    def _getneighbors(self, rowno, colno):
        neighbors = []
    
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                elif -1 < (rowno + i) < self._gridsize and -1 < (colno + j) < self._gridsize:
                    neighbors.append((rowno + i, colno + j))

        return neighbors
    
    def _getmines(self, grid, start):
        mines = []
        neighbors = self._getneighbors(*start)
    
        for i in range(self._numberofmines):
            cell = self._getrandomcell(grid)
            while cell == start or cell in mines or cell in neighbors:
                cell = self._getrandomcell(grid)
            mines.append(cell)
    
        return mines
    
    def _getnumbers(self, grid):
        for rowno, row in enumerate(grid):
            for colno, cell in enumerate(row):
                if cell != 'X':
                    values = [grid[r][c] for r, c in self._getneighbors(rowno, colno)]
                    grid[rowno][colno] = str(values.count('X'))
        return grid
    
    def _showcells(self, rowno, colno):
        if self._currentGrid[rowno][colno] != ' ':
            return 0

        self._currentGrid[rowno][colno] = self._grid[rowno][colno]
    
        if self._grid[rowno][colno] == '0':
            sum = 1
            for r, c in self._getneighbors(rowno, colno):
                if self._currentGrid[r][c] != 'F':
                    sum += self._showcells(r, c)
            return sum
        else:
            return 1

    def _playagain(self):
        choice = input('Play again? (y/n): ')
        return choice.lower() == 'y'
    
    def _parseinput(self, inputstring:str):
        pattern = r'([a-{}])([0-9]+)(f?)'.format(ascii_lowercase[self._gridsize - 1])
        validinput = re.match(pattern, inputstring)
    
        if validinput:
            rowno = int(validinput.group(2)) - 1
            colno = ascii_lowercase.index(validinput.group(1))
            flag = bool(validinput.group(3))
    
            if -1 < rowno < self._gridsize:
                return {'cell': (rowno, colno), 'flag': flag, 'isSuccessful': True}
    
        return {'cell': None, 'flag': None, 'isSuccessful': False}