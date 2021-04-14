import os
import glob
import cv2


def get_spacial_format_files(dir_path, suffix):
    """
    获取文件夹下特定后缀文件
    @param dir_path: 文件夹路径
    @param suffix: 文件后缀
    @return:
    """
    filesList = glob.glob(os.path.join(dir_path, "*" + suffix))
    return filesList


def get_train_valid_txt():
    """
    生成数据集训练与验证文件
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    labels_base_path = 'E:/personal-code/data/coco2014/labels/train2014/'
    txt_path = base_dir + '/train.txt'
    write_bboxes = True

    if os.path.exists(txt_path):
        os.remove(txt_path)

    labels = get_spacial_format_files(labels_base_path, '.txt')
    for idx, label in enumerate(labels):
        label = '/'.join(label.split('\\'))
        jpg = label.replace('.txt', '.jpg')
        jpg = jpg.replace('labels', 'images')

        img = cv2.imread(jpg)

        with open(txt_path, 'a') as txt:
            txt.write(jpg.replace('E:/personal-code/','') + ' ')

        if write_bboxes:
            hight, width, channels = img.shape
            with open(label, 'r') as file:
                lines = file.readlines()
                for idx, line in enumerate(lines):
                    paras = line.split(' ')
                    classes = int(paras[0])
                    x = float(paras[1]) * width
                    y = float(paras[2]) * hight
                    w = float(paras[3]) * width
                    h = float(paras[4]) * hight
                    if idx == len(lines) - 1:
                        string = str(int(x - w / 2)) + ',' + str(int(y - h / 2)) + ',' + str(int(x + w / 2)) + ',' + str(
                            int(y + h / 2)) + ',' + str(int(classes)) + '\n'
                    else:
                        string = str(int(x - w / 2)) + ',' + str(int(y - h / 2)) + ',' + str(int(x + w / 2)) + ',' + str(
                            int(y + h / 2)) + ',' + str(int(classes)) + ' '
                    with open(txt_path, 'a') as txt:
                        txt.write(string)
                    cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255))
        else:
            with open(txt_path, 'a') as txt:
                txt.write('\n')
        cv2.imshow('output', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    get_train_valid_txt()
    print('finish!!!')