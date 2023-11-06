import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

# Load the input image
input_image = cv2.imread("butterfly.jpg")

# Define parameters
population_size = 500
max_generations = 500
mutation_rate = 0.1
num_squares = 1000
square_size = 3

# Function to create a blurred square with color from the input image
def random_blurred_square(input_image, square_size):
    image_height, image_width, _ = input_image.shape
    x = np.random.randint(0, image_width - square_size)
    y = np.random.randint(0, image_height - square_size)

    # Extract the color from the input image
    color = input_image[y:y+square_size, x:x+square_size]

    # Apply a blur effect to the color
    color = cv2.GaussianBlur(color, (15, 15), 0)

    return (x, y, square_size, color)

# Function to draw an individual with N squares
def draw_individual(individual, image_width, image_height):
    img = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(img)
    for square in individual:
        x, y, size, color = square
        color_image = Image.fromarray(color)
        img.paste(color_image, (x, y))
    return img

# Function to calculate fitness
def calculate_fitness(individual, input_image):
    individual_image = draw_individual(individual, input_image.shape[1], input_image.shape[0])
    diff = input_image - np.array(individual_image)
    diff = diff / 255.0  # Normalize the difference array
    num_pixels = input_image.shape[0] * input_image.shape[1]
    fitness = -np.sum(diff**2) / num_pixels
    return fitness


# Initialize the population
population = [None] * population_size
for i in range(population_size):
    individual = [random_blurred_square(input_image, square_size) for _ in range(num_squares)]
    population[i] = individual

# Main GA loop
for generation in range(max_generations):
    # Evaluate fitness for each individual
    fitness_scores = [calculate_fitness(individual, input_image) for individual in population]

    # Select the top-performing individuals
    selected_indices = np.argsort(fitness_scores)[-population_size // 2:]
    selected_population = [population[i] for i in selected_indices]

    # Create a new population through crossover and mutation
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.choices(selected_population, k=2)
        crossover_point = random.randint(1, num_squares - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, num_squares - 1)
            child[mutation_point] = random_blurred_square(input_image, square_size)
        new_population.append(child)

    population = new_population

# Display the best individual
best_individual = max(population, key=lambda x: calculate_fitness(x, input_image))
best_image = draw_individual(best_individual, input_image.shape[1], input_image.shape[0])
best_image.show()
