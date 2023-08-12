import cv2
import numpy as np
import random

def simulate_grid(grid, num_steps):
    MAX_X, MAX_Y = grid.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    kernel[1][1] = 0

    for _ in range(num_steps):
        conv_res = cv2.filter2D(grid, -1, kernel)
        ret, bigger_1 = cv2.threshold(conv_res, 1, 1, cv2.THRESH_BINARY)
        ret, bigger_2 = cv2.threshold(conv_res, 2, 1, cv2.THRESH_BINARY)
        ret, smaller_4 = cv2.threshold(conv_res, 3, 1, cv2.THRESH_BINARY_INV)
        step_1 = cv2.bitwise_and(bigger_1, grid)
        step_2 = cv2.bitwise_or(step_1, bigger_2)
        step_3 = cv2.bitwise_and(step_2, smaller_4)
        grid = step_3.copy()

    return grid

def main():
    MAX_X = 512
    MAX_Y = 512

    def reset_grid():
        nonlocal grid
        for i in range(1, MAX_X - 1):
            for j in range(1, MAX_Y - 1):
                if random.randint(0, 2) == 1:
                    grid[i][j] = 1

    # Initialize the grid with zeros
    grid = np.zeros((MAX_X, MAX_Y), np.uint8)

    reset_grid()

    perform_simulation = False  # Flag to indicate if the simulation should be performed

    while True:
        if perform_simulation:
            grid = simulate_grid(grid, 10)
            perform_simulation = False

        # Display the image
        show_im = grid * 255
        cv2.imshow('sample image', show_im)

        # Wait for a key event
        k = cv2.waitKey(10) & 0XFF

        # Press space bar (key code 32) to reset the grid
        if k == 32:
            reset_grid()

        # Press 'q' (key code 113) to toggle the simulation
        if k == 113:
            perform_simulation = not perform_simulation

        # Press 'esc' (key code 27) to exit the loop
        if k == 27:
            break

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
