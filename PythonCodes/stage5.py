import cv2
import numpy as np

binary_image = cv2.imread(r'C:\Users\Interview\Desktop\IMG_8036.jpg', cv2.IMREAD_GRAYSCALE)

_, binary_threshold = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def label_character(character_contour):
    x, y, w, h = cv2.boundingRect(character_contour)

    aspect_ratio = w / h

    area = cv2.contourArea(character_contour)
    perimeter = cv2.arcLength(character_contour, True)
    
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0

    horizontal_position = "Vertical" if w < h else "Horizontal"

    if aspect_ratio < 0.5:
        label = "i"
    elif aspect_ratio > 2:
        label = "w"
    elif circularity > 0.7:
        label = "o"
    elif horizontal_position == "Horizontal":
        label = "t"
    else:
        label = "unknown"

    return label

for contour in contours:
    character_label = label_character(contour)

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(binary_image, (x, y), (x + w, y + h), 255, 2)
    cv2.putText(binary_image, character_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

cv2.imshow('Character Detection and Labeling', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
