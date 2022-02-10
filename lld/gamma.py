import cv2
import os
from tqdm import tqdm
import numpy as np

files = os.listdir('./segment_another/Normal_both/train/BR/images')
result_dir = './segment_another/Normal_both/train/BR/images/'

for file in tqdm(files):
    img = cv2.imread('./segment_another/Normal_both/train/BR/images/'+file, cv2.IMREAD_GRAYSCALE)
    histo = cv2.equalizeHist(img)

    '''

    g = 0.5
    out = img.astype(np.float)
    out = ((out / 255) ** (1 / g)) * 255
    out = out.astype(np.uint8)
    
    '''



    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # CLAHE 객체에 원본 이미지 입력하여 CLAHE가 적용된 이미지 생성
    gray_cont_dst = clahe.apply(img)
    

    cv2.imwrite(result_dir+file, gray_cont_dst)
