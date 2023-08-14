
import numpy as np
import numba
import cv2

class Board:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size[0], size[1]), dtype=np.uint8)
        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.kernel[1][1] = 0

    def randomize_board(self, density=0.5):
        """
        Randomizes the board with a given density (density is the percentage of cells that are alive)
        :param density: The density of the board
        :return: None
        """
        self.grid = np.random.choice(np.array([0, 1], dtype=np.uint8), size=self.size, p=[1 - density, density])

    def reset_board(self):
        """
        Resets the board to all dead cells
        :return: None
        """
        self.grid = np.zeros((self.size[0], self.size[1]), dtype=np.uint8)

    def step(self, n):
        """
        Steps the board n times
        :param n: The number of steps to take
        :return: None
        """
        grid = self.grid.copy()
        for _ in range(n):
            conv_res = cv2.filter2D(grid, -1, self.kernel)
            ret, bigger_1 = cv2.threshold(conv_res, 1, 1, cv2.THRESH_BINARY)
            ret, bigger_2 = cv2.threshold(conv_res, 2, 1, cv2.THRESH_BINARY)
            ret, smaller_4 = cv2.threshold(conv_res, 3, 1, cv2.THRESH_BINARY_INV)
            step_1 = cv2.bitwise_and(bigger_1, grid)
            step_2 = cv2.bitwise_or(step_1, bigger_2)
            step_3 = cv2.bitwise_and(step_2, smaller_4)
            grid = step_3.copy()

        self.grid = grid

    def __getitem__(self, key):
        print("DEBUG:" + str(key))
        return self.grid[key]

    def __setitem__(self, key, value):
        self.grid[key] = value
