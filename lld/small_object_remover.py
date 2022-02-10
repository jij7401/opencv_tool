import os
import cv2
import numpy as np

def small_object_remover(img):
    new = cv2.imread(img)
    #new = new[:, :, ::-1].copy()
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    new[new < new.mean()] = 0
#    img_pix_val = new/255
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(new, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = max(sizes)
    image = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            image[output == i + 1] = 255
#            image[img_pix_val == 1] = 255
    cv2.imwrite(img[:-4]+'_removal.png',image)

if __name__ == '__main__':

    files = os.listdir('./resize_result')

    for file in files:
        small_object_remover('./resize_result/{}'.format(file))
