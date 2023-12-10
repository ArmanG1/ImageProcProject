import cv2
import numpy as np
from matplotlib import pyplot as plt

cropped_binary_handwriting = cv2.imread(r'C:\Users\Interview\Desktop\IMG_8036.jpg', cv2.IMREAD_GRAYSCALE) 

_, binary_handwriting = cv2.threshold(cropped_binary_handwriting, 127, 255, cv2.THRESH_BINARY_INV)

lines = cv2.HoughLinesP(binary_handwriting, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

lines_image = np.copy(cropped_binary_handwriting)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)

plt.subplot(1, 3, 1), plt.imshow(cropped_binary_handwriting, cmap='gray'), plt.title('Cropped Binary Handwriting')
plt.subplot(1, 3, 2), plt.imshow(binary_handwriting, cmap='gray'), plt.title('Binary Handwriting with Lines')
plt.subplot(1, 3, 3), plt.imshow(lines_image, cmap='gray'), plt.title('Detected Lines')
plt.show()

if lines is not None:
    angles = []
    lengths = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        angles.append(angle)
        lengths.append(length)

    median_angle = np.median(angles)
    median_length = np.median(lengths)

    print(f"Median Angle of Lines: {median_angle:.2f} degrees")
    print(f"Median Length of Lines: {median_length:.2f} pixels")
