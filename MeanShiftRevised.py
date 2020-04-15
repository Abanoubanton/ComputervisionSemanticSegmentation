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
    img = cv2.imread(f1)
    sample = cv2.resize(img, (200, 200))

    sample = np.reshape(sample, (-1, 3))

    bandwidth = estimate_bandwidth(sample, quantile=.1,n_samples=100)

    ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)

    ms.fit(sample)
    labels = ms.labels_
    segmentedImg = np.reshape(labels, [200, 200])

    cluster_centers = ms.cluster_centers_

    plt.figure()
    plt.grid()
    plt.imshow(segmentedImg), plt.axis('off')
    plt.show()