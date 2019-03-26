import random
import re
import time
from string import ascii_lowercase
from enum import Enum

class GameState(Enum):
    NEW = 1
    PLAYING = 2
    ENDED = 3

class MineSweeper:
    state = GameState.NEW
    _gridsize = 0
    _currentGrid = []
    _grid = []
    _flags = []
    _starttime = 0

    def new_game(self, gridsize:int, numberofmines:int):
        self._gridsize = gridsize
        self._currentGrid = [[' ' for i in range(gridsize)] for i in range(gridsize)]
        self._grid = []
        self._flags = []
        self._starttime = 0
        self.State = GameState.NEW

    def play_user_game(self):
        if(self.state != GameState.NEW):
            print('Game state is invalid')
            return

        helpmessage = ("Type the column followed by the row (eg. a5). "
                       "To put or remove a flag, add 'f' to the cell (eg. a5f).")
        self._showgrid(self._currentGrid)
        print(helpmessage + " Type 'help' to show this message again.\n")
    
        while self.state != GameState.ENDED:
            minesleft = numberofmines - len(_flags)
            prompt = input('Enter the cell ({} mines left): '.format(minesleft))
            if inputstring == 'help':
                message = helpmessage
            else:
                self.GameStep(prompt)
                # message = "Invalid cell.  " + helpmessage

                # print('\n\n')


    def game_step(self, move:str):
        parse_result = parseinput(move)
        
        if parse_result['isSuccessful']:
            rowNo, colNo = parse_result['cell']
    
            currcell = _currentGrid[rowno][colno]
            flag = parse_result['flag']
    
            if self.State == GameState.NEW:
                _grid, mines = setupgrid(gridsize, cell, numberofmines)
                self._starttime = time.time()
                self.State = GameState.PLAYING
    
            if flag:
                if currcell == ' ':
                    _currentGrid[rowno][colno] = 'F'
                    _flags.append(cell)
                elif currcell == 'F':
                    _currentGrid[rowno][colno] = ' '
                    _flags.remove(cell)
                else:
                    message = 'Cannot put a flag there'
    
            # If there is a flag there, show a message
            elif cell in _flags:
                message = 'There is a flag there'
    
            elif _grid[rowno][colno] == 'X':
                print('Game Over\n')
                showgrid(_grid)
                if playagain():
                    playgame()
                return
    
            elif currcell == ' ':
                showcells(_grid, _currentGrid, rowno, colno)
    
            else:
                message = "That cell is already shown"
    
            if set(_flags) == set(mines):
                minutes, seconds = divmod(int(time.time() - _starttime), 60)
                print('You Win. '
                    'It took you {} minutes and {} seconds.\n'.format(minutes,
                                                                      seconds))
                showgrid(_grid)
                if playagain():
                    playgame()
                return
    
        showgrid(_currentGrid)
        print(message)

    def _setupgrid(self, gridsize, start, numberofmines):
        emptygrid = [['0' for i in range(gridsize)] for i in range(gridsize)]
    
        mines = getmines(emptygrid, start, numberofmines)
    
        for i, j in mines:
            emptygrid[i][j] = 'X'
    
        grid = getnumbers(emptygrid)
    
        return (grid, mines)
    
    def _showgrid(self, grid):
        gridsize = len(grid)
    
        horizontal = '   ' + (4 * gridsize * '-') + '-'
    
        # Print top column letters
        toplabel = '     '
    
        for i in ascii_lowercase[:gridsize]:
            toplabel = toplabel + i + '   '
    
        print(toplabel + '\n' + horizontal)
    
        # Print left row numbers
        for idx, i in enumerate(grid):
            row = '{0:2} |'.format(idx + 1)
    
            for j in i:
                row = row + ' ' + j + ' |'
    
            print(row + '\n' + horizontal)
    
        print('')
    
    def _getrandomcell(self, grid):
        gridsize = len(grid)
    
        a = random.randint(0, gridsize - 1)
        b = random.randint(0, gridsize - 1)
    
        return (a, b)
    
    def _getneighbors(self, grid, rowno, colno):
        gridsize = len(grid)
        neighbors = []
    
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                elif -1 < (rowno + i) < gridsize and -1 < (colno + j) < gridsize:
                    neighbors.append((rowno + i, colno + j))
    
        return neighbors
    
    def _getmines(self, grid, start, numberofmines):
        mines = []
        neighbors = getneighbors(grid, *start)
    
        for i in range(numberofmines):
            cell = getrandomcell(grid)
            while cell == start or cell in mines or cell in neighbors:
                cell = getrandomcell(grid)
            mines.append(cell)
    
        return mines
    
    def _getnumbers(self, grid):
        for rowno, row in enumerate(grid):
            for colno, cell in enumerate(row):
                if cell != 'X':
                    # Gets the values of the neighbors
                    values = [grid[r][c] for r, c in getneighbors(grid,
                                                                  rowno, colno)]
    
                    # Counts how many are mines
                    grid[rowno][colno] = str(values.count('X'))
    
        return grid
    
    def _showcells(self, grid, currgrid, rowno, colno):
        # Exit function if the cell was already shown
        if currgrid[rowno][colno] != ' ':
            return
    
        # Show current cell
        currgrid[rowno][colno] = grid[rowno][colno]
    
        # Get the neighbors if the cell is empty
        if grid[rowno][colno] == '0':
            for r, c in getneighbors(grid, rowno, colno):
                # Repeat function for each neighbor that doesn't have a flag
                if currgrid[r][c] != 'F':
                    showcells(grid, currgrid, r, c)

    
                    
    
    
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
    
            if -1 < rowno < gridsize:
                return {'cell': (rowno, colno), 'flag': flag, 'isSuccessful': True}
    
        return {'cell': None, 'flag': None, 'isSuccessful': False}