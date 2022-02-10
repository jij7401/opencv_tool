import cv2
import os
import numpy as np
from tqdm import tqdm

train_list = os.listdir('./detection_dataset/train/images')
test_list = os.listdir('./detection_dataset/test/images')
val_list = os.listdir('./detection_dataset/val/images')

print(len(train_list))
print(len(test_list))
print(len(val_list))
class_names = ['0', 'TR', '2', '3']
seg_names = ['TL', 'TR', 'BL', 'BR']

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def find_point(filename, mode):
    label_list = []
    for class_name in class_names:
        mask_path = './mask/{}/{}'.format(class_name, mode)
        mask_list = os.listdir(mask_path)
        count = 0

        for mask in mask_list:
            if filename[0:-4] == mask[0:-6]:
                img = cv2.imread(mask_path + '/' + mask)

                h, w, c = img.shape

                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img_binary = cv2.threshold(img2, 127, 255, 0)

                contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                contour_list = list(contours)
                max_index = 0

                for i, contour in enumerate(contour_list):
                    if (max_index < len(contour)):
                        max_index = i

                temp_contour = np.array(contour_list[max_index])
                point_count, _, _ = temp_contour.shape

                main_contour = temp_contour.reshape((point_count, 2))

                argmax = np.argmax(main_contour, axis=0)
                argmin = np.argmin(main_contour, axis=0)

                max_width = main_contour[argmax][0][0]
                max_height = main_contour[argmax][1][1]
                min_width = main_contour[argmin][0][0]
                min_height = main_contour[argmin][1][1]

                x1 = min_width - 50
                y1 = min_height - 50
                x2 = max_width + 50
                y2 = max_height + 50

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > w:
                    x2 = w
                if y2 > h:
                    y2 = h

                '''
                left_top = [x1, y1]
                right_bottom = [x2, y2]

                rect_img = cv2.rectangle(img, left_top, right_bottom, (0, 255, 255), 10)
                cv2.imwrite('./{}_{}.jpg'.format(mask, count), rect_img)
                count = count + TR
                '''

                label_list.append([class_name, x1, y1, x2, y2])
    return label_list

def crop_image(file, points, mode):
    img = cv2.imread('./mask/images/{}/{}'.format(mode,file))
    for i, point in enumerate(points):
        class_name = point[0]
        x1, y1, x2, y2 = point[1:5]
        #print(x1, y1, x2, y2)
        crop_img = img[y1: y2, x1: x2]
        seg_name = seg_names[i]

        cv2.imwrite('./seg_dataset/{}/{}/{}_{}.png'.format(mode, class_name, file[0:-4],seg_name), crop_img)


if __name__ == '__main__':
    dirlist = sorted_alphanumeric(train_list)

    #false_list = os.listdir('./detection_dataset/fail_list')

    for file in tqdm(dirlist):
        points = find_point(file, 'train')
        crop_image(file, points, '0')




