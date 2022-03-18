import os
import cv2
import numpy as np

def calculateDiceScore(mode):
    pred_list = os.listdir('./poly_mask_list/after/{}'.format(mode))

    pred_gt_dict = {'TL': '1', 'TR' : '2', 'BL' : '3', 'BR' : '4'}

    DC_list = []

    print('Dice Coefficient Score preparing for calculating about {}'.format(mode))

    for pred in pred_list:
        img_pred = cv2.imread('./poly_mask_list/after/{}/{}'.format(mode, pred), cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread('./mask_main/test/{}/{}_{}.png'.format(mode, pred[0:-14], pred_gt_dict[mode]), cv2.IMREAD_GRAYSCALE)

        dice = np.sum(img_pred[img_pred==img_gt] * 2.0 / (np.sum(img_pred) + np.sum(img_gt)))

        DC_list.append(dice)

    print(len(DC_list))

    print('Average Dice Score about {} : {}'.format(mode, str(sum(DC_list) / len(pred_list))))

if __name__ == '__main__':
    modes = ['TL', 'TR', 'BL', 'BR']
    for mode in modes:
        calculateDiceScore(mode)


