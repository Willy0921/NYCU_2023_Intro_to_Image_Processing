import numpy as np
import cv2
from utils import *


if __name__ == "__main__":
    img = cv2.imread("./Q1.jpg", cv2.IMREAD_GRAYSCALE)
    img = histogram_equalization(img)
    # cv2.imwrite("Q1_answer.jpg", img)
    cv2.imshow("Q1", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
