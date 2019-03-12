import numpy as np
import pandas as pd
import cv2

data = pd.read_csv('fer2013.csv')


pixels = data['pixels']

img = np.fromstring(data['pixels'][289], dtype=np.uint8, sep=" ").reshape((48, 48))
cv2.namedWindow("image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
imS = cv2.resize(img, (48, 48))                    # Resize image
cv2.imshow("image", imS)
# img_gray = cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR)
# cv2.imshow("image", img)
# print(img)

cv2.waitKey()