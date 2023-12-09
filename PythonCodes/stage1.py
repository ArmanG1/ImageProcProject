import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread(r'C:\Users\Interview\Desktop\IMG_8036.jpg')

colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('RGB Histogram')
plt.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title('Grayscale Histogram')
plt.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
cleaned_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

handwriting_mask = np.zeros_like(cleaned_image)
for contour in contours:
    cv2.drawContours(handwriting_mask, [contour], 0, (255), thickness=cv2.FILLED)

x, y, w, h = cv2.boundingRect(handwriting_mask)

cropped_handwriting = image[y:y+h, x:x+w]

cropped_gray = cv2.cvtColor(cropped_handwriting, cv2.COLOR_BGR2GRAY)

_, binary_handwriting = cv2.threshold(cropped_gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(binary_handwriting, cmap='gray')
plt.title('Extracted Handwriting')
plt.show()

plt.imshow(cleaned_image, cmap='gray')
plt.title('Cleaned Image')
plt.show()

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Contours on Cleaned Image')
plt.show()

