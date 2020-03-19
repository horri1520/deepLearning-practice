import glob
import os
import random
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator


def main():
    model = load_model('./models/model.h5')
    test_dir = './dataset/test/'
    right_num = 0
    for a in range(1000):
        print('Attempt {}/1000'.format(a+1))
        before_img = glob.glob(test_dir + 'img/*.jpg')
        os.remove(before_img[0])
        num = random.randint(0,12499)
        label_num = random.randint(0,1)
        if label_num == 0:
            label = 'cat'
        elif label_num == 1:
            label = 'dog'
        target = glob.glob('../CNNelements/dogsvscats/train_imgs/' + label + '.' + str(num) + '.jpg')
        shutil.copy(target[0], test_dir + 'img')
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150,150),
            batch_size=20,
            class_mode=None
        )
        pred = model.predict_generator(
            test_generator,
            steps=1,
            verbose=1)

        for i in pred:
            cat_per = 1-i
            dog_per = i
            if cat_per < dog_per:
                result = 'dog'
            else:
                result = 'cat'
            if result == label:
                judge = 'RIGHT'
                right_num = right_num + 1
            else:
                judge = 'WRONG'
            print("predict: {}  answer: {} {}".format(result, label, judge))
    right_per = right_num/1000*100
    print("Correct answer rate: {}%".format(right_per))


if __name__ == "__main__":
    main()
