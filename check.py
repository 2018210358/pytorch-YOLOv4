import os
import sys
import glob

import cv2


if __name__ == '__main__':
    file_path = 'data/train.txt'
    f = open(file_path, 'r')
    lines = f.readlines()
    num = 0
    for line in lines:
        img_path = line.split(' ')[0]
        status = os.path.exists(img_path)
        if not status:
            num += 1
            print(img_path)
    print(str(num) + 'images are missing!!!')
    print('Finish!!!')
