import numpy as np
import pandas as pd
import cv2

data = pd.read_csv('fer2013.csv')


pixels = data['pixels']

img = np.fromstring(data['pixels'][0], dtype=np.uint8, sep=" ").reshape((48, 48))
# img_gray = cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR)
cv2.imshow("image", img)
# print(img)

cv2.waitKey()