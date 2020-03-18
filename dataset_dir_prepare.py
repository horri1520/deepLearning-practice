import os

def main():
    base_dir = './dataset/'
    os.mkdir(base_dir)
    sub_dir_name = ['train', 'validation','test']
    for i in sub_dir_name:
        os.mkdir(base_dir + i)
        labels = ['dogs', 'cats']
        os.mkdir(base_dir + i + '/' + labels[0])
        os.mkdir(base_dir + i + '/' + labels[1])


if __name__ == "__main__":
    main()