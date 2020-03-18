from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import h5py


dataset_dir = './dataset/'

train_dir = dataset_dir + 'train/'
validation_dir = dataset_dir + 'validation/'


def pre_process():
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )
    for data,label in train_generator:
        print(data.shape)
        print(label.shape)
        break
    for data,label in validation_generator:
        print(data.shape)
        print(label.shape)
        break
    create_model()


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))

    model.summary()

    model.compile(loss="binary_crossentropy",
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=["acc"])
    learning(model)


def learning(model):
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=50)
    



if __name__ == "__main__":
    pre_process()