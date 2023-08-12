from util import read_image_to_numpy_array, show_images, write_results_to_csv
from tqdm import tqdm
import numpy as np
import random
import numba
import cv2

MAX_X = 128
MAX_Y = 128
STEP_AMOUNT = 1250
INITIAL_POPULATION_SIZE = 100
TARGET_IMAGE_PATH = "./targets/hollow-square.png"
ITERATION_COUNT = 10000
MUTATION_RATE = 0.75

CROSSOVER_METHOD = "point_crossover"
MUTATION_METHOD = "point_mutate"

DATA_COLECTION_PERIOD = ITERATION_COUNT // 10

def get_max_fitness(target_image):
    return np.sum(target_image)

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


@numba.jit(nopython=True)
def generate_genome(width, height):
    genome = np.zeros((width, height), np.uint8)
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if random.randint(0, 2) == 1:
                genome[i][j] = 1
    return genome


@numba.jit(nopython=True)
def point_mutate_genome(genome, mutation_rate):
    for i in range(genome.shape[0]):
        for j in range(genome.shape[1]):
            if random.random() < mutation_rate:
                genome[i][j] = 1 - genome[i][j]
    return genome


@numba.jit(nopython=True)
def point_crossover_genomes(genome_1, genome_2):
    child_genome = np.zeros(genome_1.shape, np.uint8)
    for i in range(genome_1.shape[0]):
        for j in range(genome_1.shape[1]):
            if random.random() < 0.5:
                child_genome[i][j] = genome_1[i][j]
            else:
                child_genome[i][j] = genome_2[i][j]
    return child_genome


def calculate_fitness(genome, target_image):
    grid = np.zeros((MAX_X, MAX_Y), np.uint8)
    grid[MAX_X // 2 - 16:MAX_X // 2 + 16, MAX_Y // 2 - 16:MAX_Y // 2 + 16] = genome
    grid = simulate_grid(grid, STEP_AMOUNT)
    return np.sum(grid * target_image)

def get_genome_image(genome, n):
    grid = np.zeros((MAX_X, MAX_Y), np.uint8)
    grid[MAX_X // 2 - 16:MAX_X // 2 + 16, MAX_Y // 2 - 16:MAX_Y // 2 + 16] = genome
    grid = simulate_grid(grid, n)
    return grid


def main():

    target_image = read_image_to_numpy_array(TARGET_IMAGE_PATH)

    initial_population = [generate_genome(32, 32) for _ in range(INITIAL_POPULATION_SIZE)]
    fitnesses = [calculate_fitness(genome, target_image) for genome in initial_population]

    def save_results_to_csv(show_image, at_iteration):
        nonlocal initial_population, fitnesses
        best_genome = initial_population[np.argmax(fitnesses)]
        best_genome_images = [get_genome_image(best_genome, 0),
                              get_genome_image(best_genome, 250),
                              get_genome_image(best_genome, 500),
                              get_genome_image(best_genome, 1250),
                              get_genome_image(best_genome, 1500)]

        best_genome_fitness = np.max(fitnesses)
        best_genome_succes_percentage = best_genome_fitness / get_max_fitness(target_image) * 100
        print(f"Best genome fitness: {best_genome_fitness} (which is {best_genome_succes_percentage:.3f}%)")
        STEP_AMOUNT = 2000
        print(f"Future fitness: {calculate_fitness(best_genome, target_image)}")

        if show_image:
            show_images(best_genome_images)

        write_results_to_csv(best_genome,
                             best_genome_fitness,
                             get_max_fitness(target_image),
                             best_genome_succes_percentage,
                             MAX_X,
                             MAX_Y,
                             at_iteration,
                             STEP_AMOUNT,
                             INITIAL_POPULATION_SIZE,
                             TARGET_IMAGE_PATH,
                             MUTATION_RATE,
                             MUTATION_METHOD,
                             CROSSOVER_METHOD,
                             )

    # Evolution
    for i in tqdm(range(ITERATION_COUNT)):
        parents = random.choices(initial_population, weights=fitnesses, k=2)

        child = point_crossover_genomes(*parents)
        child = point_mutate_genome(child, MUTATION_RATE)
        child_fitness = calculate_fitness(child, target_image)

        if child_fitness > min(fitnesses):
            min_index = np.argmin(fitnesses)
            initial_population[min_index] = child
            fitnesses[min_index] = child_fitness

        if i % DATA_COLECTION_PERIOD == 0:
            save_results_to_csv(show_image=False, at_iteration=i)

    save_results_to_csv(show_image=True, at_iteration=ITERATION_COUNT)


if __name__ == '__main__':
    main()
