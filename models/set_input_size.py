import os
import zipfile
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SET_IMAGES_IN_CLASS = 2000
NO_CLASSES = 8

# Ensure that everything in the 'dataset' directory ends with '.png'
print("cleaning up extensions from last run...")
for subdir, dirs, files in os.walk("/home/kta12/final-project/data_generator_output/"):
    for filename in files:
        if filename.endswith('.csv.png'):
            filePath = os.path.join(subdir, filename)
            newFilePath = filePath[:-4]
            os.rename(filePath, newFilePath)
        elif filename.endswith('.csv'):
            pass
        elif not filename.endswith('.png'):
            filePath = os.path.join(subdir, filename)
            newFilePath = filePath + '.png'
            os.rename(filePath, newFilePath)

# We need to match the number of input images that the user specified with SET_IMAGES_IN_CLASS
# Unfortunately, TensorFlow gives us no easy way to do this
# We instead just rip the '.png' extension right off of the correct number of images for each class
# Removing the extension means the image won't load, effectively letting us control input size
print("fixing up extensions for this run...")

def match_num_images(path):
    num_matched = 0
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.png'):
                num_matched += 1
                if num_matched > SET_IMAGES_IN_CLASS:
                    filePath = os.path.join(subdir, filename)
                    newFilePath = filePath[:-4]
                    os.rename(filePath, newFilePath)

for classNumber in [0, 1, 2, 3, 4, 5, 6, 7]:
    match_num_images("/home/kta12/final-project/data_generator_output/class" + str(classNumber) + "/")
