import tensorflow as tf
import keras
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import  img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model # import load_model function to load trained CNN model for Sign classification



print(tf.__version__)
print(keras.__version__)
Training_CNN = True
NUM_CATEGORIES = 0

def load_data(data_dir):
    '''
    Loading data from Train folder.
    
    Returns a tuple `(images, labels)` , where `images` is a list of all the images in the train directory,
    where each image is formatted as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. 
    `labels` is a list of integer labels, representing the categories for each of the
    corresponding `images`.
    '''
    global NUM_CATEGORIES
    images = list()
    labels = list()
    for category in range(NUM_CATEGORIES):
        categories = os.path.join(data_dir, str(category))
        for img in os.listdir(categories):
            img = load_img(os.path.join(categories, img), target_size=(30, 30))
            image = img_to_array(img)
            images.append(image)
            labels.append(category)
    
    return images, labels

def train_SignsModel(data_dir,IMG_HEIGHT = 30,IMG_WIDTH = 30,EPOCHS = 50, save_model = False, saved_model = "/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/saved_model5.h5"):
    
    
    train_path = data_dir
    global NUM_CATEGORIES

    # Number of Classes
    NUM_CATEGORIES = len(os.listdir(train_path))
    print("NUM_CATEGORIES = " , NUM_CATEGORIES)

    # Visualizing all the different Signs
    img_dir = pathlib.Path(train_path)
    plt.figure(figsize=(14,14))
    index = 0
    for i in range(NUM_CATEGORIES):
        plt.subplot(7, 7, i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        print(img_dir)
        sign = list(img_dir.glob(f'{i}/*'))[0]
        img = load_img(sign, target_size=(IMG_WIDTH, IMG_HEIGHT))
        plt.imshow(img)
    plt.show()

    images, labels = load_data(train_path)
    print(len(labels))
    # One hot encoding the labels
    labels = to_categorical(labels)
    
    # Splitting the dataset into training and test set
    # 70% train, 15% val, 15% test
    x_train, x_temp, y_train, y_temp = train_test_split(np.array(images), labels, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)


    #========================================= Model Creation ===============================================
    #========================================================================================================
    model = Sequential()
    # First Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    # Second Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    # Flattening the layer and adding Dense Layer
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))
    model.summary()
    #========================================================================================================
    # The optimizer determines how the model updates its weights based on the computed loss during training.
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    # Compiling the model
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    # Fitting the model
    history = model.fit(x_train, 
                        y_train,
                        validation_data = (x_val, y_val), 
                        epochs=EPOCHS, 
                        steps_per_epoch=60)

    print(x_test.shape)
    print(y_test.shape)
    print(y_test)
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test set accuracy: ', accuracy * 100)

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    #========================================= Saving Model =================================================
    #========================================================================================================
    # save model and architecture to single file
    if save_model:
        model.save(saved_model)
        print("Saved model to disk")
    #========================================================================================================



def main():
    if Training_CNN:
        train_SignsModel("/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/dataset_signs")
    


if __name__ == '__main__':
	main()