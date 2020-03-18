def main():
    import os
    import glob

    base_dir = './dataset/'
    imgs = glob.glob(base_dir + '*/*/*.jpg')
    print(imgs)
    for i in imgs:
        os.remove(i)


if __name__ == "__main__":
    main()