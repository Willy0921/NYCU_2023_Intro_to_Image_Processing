import numpy as np
import cv2
from utils import *


if __name__ == "__main__":
    img1 = cv2.imread("./Q1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./Q2.jpg", cv2.IMREAD_GRAYSCALE)

    img = histogram_specification(img1, img2)
    # cv2.imwrite("Q2_answer.jpg", img)

    cv2.imshow("Q2", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
