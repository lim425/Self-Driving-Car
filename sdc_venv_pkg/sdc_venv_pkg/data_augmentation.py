from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import numpy as np
import os
from PIL import Image
import cv2

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    # horizontal_flip=True,
    fill_mode='constant', cval=125

)

image_directory = "/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/dataset_signs/4/"
size = 30
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((size,size))
        dataset.append(np.array(image))

x = np.array(dataset)

i=0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir='/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/data_augmented/',
                          save_prefix='aug',
                          save_format='png'):
    i += 1
    if i > 10:
        break