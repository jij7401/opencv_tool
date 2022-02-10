import cv2
import os
from tqdm import tqdm

files = os.listdir('./segment_another/Normal_both/val/TL/images')
result_dir = './main_histogram/Normal_both/val/images/'

for file in tqdm(files):
    img = cv2.imread('./segment_another/Normal_both/val/TL/images/'+file, cv2.IMREAD_GRAYSCALE)

    histo = cv2.equalizeHist(img)


    cv2.imwrite(result_dir+file, histo)
