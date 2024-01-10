import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.optimizers.legacy import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

print("Imports Loaded")


path = "./archive/traffic_Data/DATA/"
labelfile = "./archive/labels.csv"
batch_size_val = 16
steps_per_epoch_val = 50
epochs_val = 20
imgDims = (100, 100, 3)
testratio = 0.1
validationratio = 0.1

print("Global Constants Set")


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, imgDims[:2])
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img



def mymodel():
    nooffilters = 60
    sizeoffilters = (5, 5)
    sizeoffilters2 = (3, 3)
    sizeofpool = (2, 2)
    noofnodes = 500
    model = Sequential()
    model.add(
        (
            Conv2D(
                nooffilters, sizeoffilters, input_shape=(100, 100, 1), activation="relu"
            )
        )
    )
    model.add((Conv2D(nooffilters, sizeoffilters, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeofpool))

    model.add((Conv2D(nooffilters // 2, sizeoffilters2, activation="relu")))
    model.add((Conv2D(nooffilters // 2, sizeoffilters2, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeofpool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noofnodes, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(noofclasses, activation="softmax"))

    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    count = 0
    images = []
    classno = []
    mylist = os.listdir(path)
    print("Total Classes Detected: ", len(mylist))
    noofclasses = 10 # len(mylist)
    print("Importing Classes .....")
    for i in range(noofclasses):
        mypics = os.listdir(path + str(count))
        for y in mypics:
            current = cv2.imread(path + "/" + str(count) + "/" + y)
            # Ensure the image is not None (i.e., reading was successful)
            if current is not None:
                # Resize the image to the desired dimensions
                current = cv2.resize(current, imgDims[:2])
                images.append(current)
                classno.append(count)
        print(str(count + 1) + "/" + str(noofclasses))
        count = count + 1

    # print(images)
    images = np.array(images)
    classno = np.array(classno)

    X_train, X_test, Y_train, Y_test = train_test_split(
        images, classno, test_size=testratio
    )
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=validationratio
    )

    print("DATA SHAPES")
    print("Train:  ")
    print(X_train.shape, Y_train.shape)
    print("Validation:   ")
    print(X_validation.shape, Y_validation.shape)
    print("Test:  ")
    print(X_test.shape, Y_test.shape)

    data = pd.read_csv(labelfile)
    print("data_shape", data.shape, type(data))
    
    # Replace with tf.keras.utils.image_dataset_from_directory()

    X_train = np.array(list(map(preprocessing, X_train)))
    X_validation = np.array(list(map(preprocessing, X_validation)))
    X_test = np.array(list(map(preprocessing, X_test)))
    # cv2.imshow("Gray Scale Images: ", X_train[random.randint(0,len(X_train)-1)])
    # cv2.waitKey(200)

    X_train = X_train.reshape(X_train.shape[0], imgDims[0], imgDims[1], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], imgDims[0], imgDims[1], 1)
    X_test = X_test.reshape(X_test.shape[0], imgDims[0], imgDims[1], 1)

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        rotation_range=10,
    )
    datagen.fit(X_train)
    batches = datagen.flow(X_train, Y_train, batch_size=batch_size_val)
    X_batch, Y_batch = next(batches)

    Y_train = to_categorical(Y_train, noofclasses)
    Y_validation = to_categorical(Y_validation, noofclasses)
    Y_test = to_categorical(Y_test, noofclasses)

    model = mymodel()
    history = model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size_val),
        steps_per_epoch=steps_per_epoch_val,
        epochs=100,
        validation_data=(X_validation, Y_validation),
        shuffle=1,
        verbose=2,
    )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test Loss: ", score[0])
    print("Test Accuracy: ", score[1])
    
    save_model(model, "model.h5")
    print("Model saved")

    # pickle_out = open("model_trained_avnmht.p", "wb")
    # pickle.dump(mymodel(), pickle_out)
    # pickle_out.close()
    # cv2.waitKey(0)
