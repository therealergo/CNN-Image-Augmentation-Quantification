import os
import zipfile
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SET_IMAGE_WIDTH = 224
SET_COLOR_MODE = 'rgb'
SET_AUGMENT_IMAGES = True
SET_NUM_EPOCHS = 150
SET_BATCH_SIZE = 64
SET_VALIDATION_PCT = 0.2
EPSILON = 2e-5
MOMENTUM = 0.9
cardinality = 32
KERNEL_REGULARIZER = 0.0001
SET_IMAGES_IN_CLASS = 2000
NO_CLASSES = 8

for SET_AUG_ROT in range(111, 120, 4):
    print("SET_AUG_ROT = " + str(SET_AUG_ROT))

    # Start timing this run
    start = time.time()

    # Build up our simple classification model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
            SET_IMAGE_WIDTH, SET_IMAGE_WIDTH, 1 if SET_COLOR_MODE == 'grayscale' else 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(8, activation=tf.nn.softmax)
    ])
    model.summary()
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0002
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Data generator used for training
    # Depending on settings, may or may not have image augmentation enabled
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=SET_VALIDATION_PCT,

        rotation_range=SET_AUG_ROT if SET_AUGMENT_IMAGES else 0
    #    width_shift_range=0.3 if SET_AUGMENT_IMAGES else 0,
    #    height_shift_range=0.3 if SET_AUGMENT_IMAGES else 0,
    #    shear_range=0.4 if SET_AUGMENT_IMAGES else 0,
    #    zoom_range=0.4 if SET_AUGMENT_IMAGES else 0,
    #    horizontal_flip=True if SET_AUGMENT_IMAGES else False
    )

    # Data generator used for validation
    # Performs no image augmentation
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=SET_VALIDATION_PCT
    )

    # Link each of our data generators up with our actual on-disk dataset
    seed = random.getrandbits(32)
    train_generator = train_datagen.flow_from_directory(
        directory="/home/kta12/final-project/data_generator_output/",
        target_size=(SET_IMAGE_WIDTH, SET_IMAGE_WIDTH),
        batch_size=SET_BATCH_SIZE,
        color_mode=SET_COLOR_MODE,
        class_mode='categorical',
        subset='training',
        seed=seed
    )
    validation_generator = test_datagen.flow_from_directory(
        directory="/home/kta12/final-project/data_generator_output/",
        target_size=(SET_IMAGE_WIDTH, SET_IMAGE_WIDTH),
        batch_size=SET_BATCH_SIZE,
        color_mode=SET_COLOR_MODE,
        class_mode='categorical',
        subset='validation',
        seed=seed
    )

    # Now, let's actually train our model
    history = model.fit(
        train_generator,
        steps_per_epoch=int(SET_IMAGES_IN_CLASS /
                            SET_BATCH_SIZE * (1.0 - SET_VALIDATION_PCT)),
        epochs=SET_NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=int(SET_IMAGES_IN_CLASS /
                             SET_BATCH_SIZE * SET_VALIDATION_PCT),
        verbose=2
    )

    # Function that lets us test an individual image against our trained model
    # Prints out the resulting class that the model gives that image


    def test_image(path):
        test_img = tf.keras.preprocessing.image.load_img(
            path=path,
            target_size=(SET_IMAGE_WIDTH, SET_IMAGE_WIDTH),
            color_mode=SET_COLOR_MODE
        )
        test_img = tf.keras.preprocessing.image.img_to_array(test_img) / 255.0
        test_img = np.expand_dims(test_img, axis=0)
        test_img_res = model.predict(test_img)
        print(path + " ===> class = " +
              str(tf.argmax(test_img_res[0], axis=0).numpy()))

    """
    # Test out a whole bunch of images, to ensure that we're seeing classes that make sense
    for classNumber in [0, 1, 2, 3, 4, 5, 6, 7]:
        test_image('./dataset/class' + str(classNumber) + '/img0.png')
        test_image('./dataset/class' + str(classNumber) + '/img1.png')
        test_image('./dataset/class' + str(classNumber) + '/img10.png')
        test_image('./dataset/class' + str(classNumber) + '/img11.png')
    test_image('./test/testclass1.png')
    test_image('./test/testclass4.png')
    test_image('./bananalamp.png')
    test_image('./PepeSmall.png')
    """

    # Read the training & validation statistics from the model's training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Save our accuracies to a file
    np.savetxt("./out/" + str(SET_AUG_ROT) + "tra_accuracy_last.csv",      acc, delimiter=",")
    np.savetxt("./out/" + str(SET_AUG_ROT) + "val_accuracy_last.csv",  val_acc, delimiter=",")
    np.savetxt("./out/" + str(SET_AUG_ROT) + "tra_loss_last.csv"    ,     loss, delimiter=",")
    np.savetxt("./out/" + str(SET_AUG_ROT) + "val_loss_last.csv"    , val_loss, delimiter=",")

    # Plot out the training & validation accuracy
    num_epochs = range(len(acc))
    plt.plot(num_epochs, acc, 'bo', label='Training accuracy')
    plt.plot(num_epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(num_epochs, loss, 'bo', label='Training Loss')
    plt.plot(num_epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    end = time.time()
    print("The time of execution of above program is :", end-start)
