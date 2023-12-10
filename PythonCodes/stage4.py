import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology

cropped_handwriting = cv2.imread(r'C:\Users\Interview\Desktop\IMG_8036.jpg', cv2.IMREAD_GRAYSCALE)

_, binary_handwriting = cv2.threshold(cropped_handwriting, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)
binary_regions = cv2.morphologyEx(binary_handwriting, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(binary_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

character_regions = np.zeros_like(binary_handwriting)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  
        cv2.drawContours(character_regions, [contour], -1, 255, thickness=cv2.FILLED)

kernel = np.ones((3, 3), np.uint8)
character_regions = cv2.morphologyEx(character_regions, cv2.MORPH_CLOSE, kernel)
character_regions = cv2.morphologyEx(character_regions, cv2.MORPH_OPEN, kernel)

skeleton = morphology.skeletonize(character_regions / 255)

label_readability = "Messy" if np.sum(skeleton) > 0 else "Clear"

plt.subplot(1, 3, 1), plt.imshow(binary_handwriting, cmap='gray'), plt.title('Binary Handwriting')
plt.subplot(1, 3, 2), plt.imshow(character_regions, cmap='gray'), plt.title('Separated Characters')
plt.subplot(1, 3, 3), plt.imshow(skeleton, cmap='gray'), plt.title(f'Baselines - {label_readability}')
plt.show()
