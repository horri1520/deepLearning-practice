import shutil
import glob

def train_imgs_prepare():
    for i in range(0,2000):
        elements_dir = '../CNNelements/dogsvscats/train_imgs/'
        base_dir = './dataset/train/'
        train_dogs = glob.glob(elements_dir + 'dog.' + str(i) + '.jpg', recursive=True)
        train_cats = glob.glob(elements_dir + 'cat.' + str(i) + '.jpg', recursive=True)
        shutil.copy(train_dogs[0], base_dir + 'dogs/')
        shutil.copy(train_cats[0], base_dir + 'cats/')
    validation_imgs_prepare()


def validation_imgs_prepare():
    for i in range(2000,3000):
        elements_dir = '../CNNelements/dogsvscats/train_imgs/'
        base_dir = './dataset/validation/'
        validation_dogs = glob.glob(elements_dir + 'dog.' + str(i) + '.jpg', recursive=True)
        validation_cats = glob.glob(elements_dir + 'cat.' + str(i) + '.jpg', recursive=True)
        shutil.copy(validation_dogs[0], base_dir + 'dogs/')
        shutil.copy(validation_cats[0], base_dir + 'cats/')
    test_imgs_prepare()


def test_imgs_prepare():
    for i in range(1,1001):
        elements_dir = '../CNNelements/dogsvscats/test_imgs/'
        base_dir = './dataset/test/'
        test_imgs = glob.glob(elements_dir + str(i) + '.jpg', recursive=True)
        shutil.copy(test_imgs[0], base_dir)


if __name__ == "__main__":
    train_imgs_prepare()