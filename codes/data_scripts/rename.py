import os
import glob


def main():
    folder = '../../results/006_RRDBNet_ILRx4_Flickr2K_100w+/DIV2K100'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
#         new_path = img_path.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')
        new_path = img_path.replace('.png', 'x4.png')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()