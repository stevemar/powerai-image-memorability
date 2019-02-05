from PIL import Image
import numpy as np
import random
import multiprocessing as mp
import csv

# Function to load data split from lamem
def load_split(split_file):
    contents = list(csv.reader(open(split_file), delimiter=" "))
    contents = [[x[0], float(x[1])] for x in contents]
    return contents

# Function to load a single image
def load_image(image_file):
    return np.array(Image.open(image_file).resize((227, 227)).convert("RGB"), dtype="float32") / 255.

# Function that yields random samples of LaMem data
def lamem_generator(split_file, batch_size):
    while True:
        random_files = random.sample(split_file, batch_size)
        inputs = mp.Pool().map(load_image, ["lamem/images/" + i[0] for i in random_files])
        final_labels = [[i[1]] for i in random_files]
        yield(np.array(inputs), np.array(final_labels))
