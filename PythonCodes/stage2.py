import cv2
import numpy as np
from matplotlib import pyplot as plt

image_removed_text = cv2.imread(r'C:\Users\Interview\Desktop\IMG_8036.jpg')

gray_removed_text = cv2.cvtColor(image_removed_text, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray_removed_text, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

if lines is not None:
    angles = [line[0][1] for line in lines]
    main_orientation = np.median(angles) * 180 / np.pi
    print(f"Main orientation of the page: {main_orientation:.2f} degrees")

blurred_removed_text = cv2.GaussianBlur(gray_removed_text, (5, 5), 0)
sobelx = cv2.Sobel(blurred_removed_text, cv2.CV_64F, 1, 0, ksize=5)

avg_letter_width = np.mean(np.sum(np.abs(sobelx), axis=0))
print(f"Average letter width: {avg_letter_width:.2f} pixels")

plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image_removed_text, cv2.COLOR_BGR2RGB)), plt.title('Image')
plt.subplot(1, 3, 2), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')
plt.subplot(1, 3, 3), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel Filter')
plt.show()
