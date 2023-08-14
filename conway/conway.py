
from board import Board
import cv2

GRID_SIZE = (512, 512)
board = Board(GRID_SIZE)

def main():

    perform_simulation = False  # Flag to indicate if the simulation should be performed

    while True:
        if perform_simulation:
            board.step(10)
            perform_simulation = False

        # Display the image
        print(board.grid.shape)
        show_im = board.grid * 255
        cv2.imshow('sample image', show_im)

        # Wait for a key event
        k = cv2.waitKey(10) & 0XFF

        match k:
            case 32:  # Spacebar
                board.randomize_board()
            case 113:  # q
                perform_simulation = not perform_simulation
            case 27:  # Escape
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
