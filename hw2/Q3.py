import numpy as np
import cv2
from utils import *


if __name__ == "__main__":
    img = cv2.imread("./Q3.jpg", cv2.IMREAD_GRAYSCALE)

    img = gaussian_filter(img, 1, 5, 25)
    print(img)

    # cv2.imwrite("./Q3_answer.jpg", img)
    cv2.imshow("Q3", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
