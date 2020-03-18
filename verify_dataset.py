import glob


def main():
    base_dir = './dataset/'
    train_dogs = glob.glob(base_dir + 'train/dogs/*.jpg', recursive=True)
    train_cats = glob.glob(base_dir + 'train/cats/*.jpg', recursive=True)
    validation_dogs = glob.glob(base_dir + 'validation/dogs/*.jpg', recursive=True)
    validation_cats = glob.glob(base_dir + 'validation/cats/*.jpg', recursive=True)
    test_dogs = glob.glob(base_dir + 'test/dogs/dog*.jpg', recursive=True)
    test_cats = glob.glob(base_dir + 'test/cats/cat*.jpg', recursive=True)
    print('train: dogs {}  cats {}\nvalidation: dogs {}  cats {}\ntest: dogs {}  cats {}'.format(len(train_dogs), len(train_cats), len(validation_dogs), len(validation_cats), len(test_dogs), len(test_cats)))


if __name__ == "__main__":
    main()