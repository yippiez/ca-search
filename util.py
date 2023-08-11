
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import csv

def read_image_to_numpy_array(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = np.array(img)
    img = img / 255
    img = img.astype(np.uint8)
    return img

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def show_images(images):
    image_count = len(images)
    fig, axes = plt.subplots(1, image_count)
    for i in range(image_count):
        axes[i].imshow(images[i], cmap='gray')
    plt.show()


"""
BEST_GENOME
BEST_GENOME_FITNESS
MAX_FITNESS
SUCCES_PERCENTAGE
MAX_X
MAX_Y
ITERATION_COUNT
SIMULATION_STEP_AMOUNT
INITIAL_POPULATION_SIZE
TARGET_IMAGE_PATH
"""
def write_results_to_csv(BEST_GENOME, BEST_GENOME_FITNESS, MAX_FITNESS, SUCCES_PERCENTAGE, MAX_X, MAX_Y, ITERATION_COUNT, SIMULATION_STEP_AMOUNT, INITIAL_POPULATION_SIZE, TARGET_IMAGE_PATH, MUTATION_RATE, MUTATION_METHOD, CROSSOVER_METHOD):  # noqa
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # if file is empty, write header

        if file.tell() == 0:
            writer.writerow(["BEST_GENOME", "BEST_GENOME_FITNESS", "MAX_FITNESS", "SUCCES_PERCENTAGE", "MAX_X", "MAX_Y", "ITERATION_COUNT", "SIMULATION_STEP_AMOUNT", "INITIAL_POPULATION_SIZE", "TARGET_IMAGE_PATH", "MUTATION_RATE", "MUTATION_METHOD", "CROSSOVER_METHOD"])  # noqa

        writer.writerow([BEST_GENOME, BEST_GENOME_FITNESS, MAX_FITNESS, SUCCES_PERCENTAGE, MAX_X, MAX_Y, ITERATION_COUNT, SIMULATION_STEP_AMOUNT, INITIAL_POPULATION_SIZE, TARGET_IMAGE_PATH, MUTATION_RATE,MUTATION_METHOD, CROSSOVER_METHOD])  # noqa
