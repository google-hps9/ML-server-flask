import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def preprocess_image(raw_image, corners= np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32), size= 224):
    
    raw_image = cv2.resize(raw_image,(960,540))

    target_corners = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)    
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    img = cv2.warpPerspective(raw_image, matrix, (size, size))
    
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (size, size))
    
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img

def show_region(raw_image, corners= np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32), size= 224):
    
    img = cv2.resize(raw_image,(960,540))
    corners = np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32)
    contours = np.array([corners], dtype=np.int32)
    cv2.polylines(img, contours, isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.imshow('Image with Trapezoid', img)
    cv2.waitKey(1)


if __name__ == "__main__" : # to adjust camera perspective
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            show_region(frame)
            cv2.imshow("preprocessed",preprocess_image(frame))