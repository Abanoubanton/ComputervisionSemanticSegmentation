import os
import cv2
import numpy as np
import glob
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt


img_path = "Images"

data_path = os.path.join(img_path,'*g')

files = glob.glob(data_path)
i = 1
data = []
for f1 in files:

    image = cv2.imread(f1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    print(pixel_values.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    # show the image
    plt.imshow(masked_image)
    plt.show()

