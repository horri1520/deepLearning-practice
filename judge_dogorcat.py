import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator


def judgement():
    model_2000 = load_model('./models/model_2000.h5')
    model_4000 = load_model('./models/model_4000.h5')
    test_dir = './dataset/test/'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode=None
    )
    pred0 = model_2000.predict_generator(
        test_generator,
        steps=1000,
        verbose=1)
    pred1 = model_4000.predict_generator(
        test_generator,
        steps=1000,
        verbose=1)
    result(pred0, pred1)


def result(pred0, pred1):
    labels = ['dog', 'cat']
    for i in pred0:
        cls = np.argmax(i)
        score = np.max(i)
        print("pred: {}  score = {:.3f}".format(labels[cls], score))

    print("-" * 30)

    for i in pred1:
        cls = np.argmax(i)
        score = np.max(i)
        print("pred: {}  score = {:.3f}".format(labels[cls], score))


if __name__ == "__main__":
    judgement()
