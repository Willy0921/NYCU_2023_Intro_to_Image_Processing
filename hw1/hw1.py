import cv2
import numpy as np
import math

def exchange_position(img, h_window, w_window):
    tmp_img = img.copy()
    img[:h_window, -w_window:, :] = tmp_img[:h_window, :w_window, :]
    img[:h_window, :w_window, :] = tmp_img[:h_window, -w_window:, :]
    return img

def grayscale(img):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            gray_value = np.sum(img[row, col]) / 3
            for i in range(3):
                img[row, col, i] = gray_value
    return img

def intensity_resolution(img):
    img = grayscale(img)

    img = np.floor_divide(img, 64)
    img = img * 64
    img = img.astype(np.uint8)

    return img

def color_filter_red(img):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if not (img[row, col, 2] > 150 
                    and img[row, col, 2] * 0.6 > img[row, col, 0] 
                    and img[row, col, 2] * 0.6 > img[row, col, 1]):
                
                gray_value = np.sum(img[row, col]) / 3
                for i in range(3):
                    img[row, col, i] = gray_value
    return img

def color_filter_yellow(img):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            d = max(img[row, col, 1], img[row, col, 2]) - min(img[row, col, 1], img[row, col, 2])
            if not ((img[row, col, 1] * 0.3 + img[row, col, 2] * 0.3) > img[row, col, 0] and d < 50):
                gray_value = np.sum(img[row, col]) / 3
                for i in range(3):
                    img[row, col, i] = gray_value
    return img

def channel_operation(img):
    img = img.astype(np.int32)
    img[:, :, 1] *= 2
    img[np.where(img[:, :, 1] > 255)][1] = 255
    img = img.astype(np.uint8)
    return img

def bilinear_interpolation(img, scale):
    height, width = img.shape[0], img.shape[1]
    blank_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for row in range(height):
        for col in range(width):
            y = row/scale
            x = col/scale
            y1, y2 = int(y), min(int(y + 1), int(height/2))
            x1, x2 = int(x), min(int(x + 1), int(width/2))

            f11, f12 = img[y1, x1], img[y1, x2]
            f21, f22 = img[y2, x1], img[y2, x2]

            fx1 = (x2-x)/(x2-x1)*f11 + (x-x1)/(x2-x1)*f12
            fx2 = (x2-x)/(x2-x1)*f21 + (x-x1)/(x2-x1)*f22

            fxy = (y2-y)/(y2-y1)*fx1 + (y-y1)/(y2-y1)*fx2

            blank_img[row, col] = fxy.astype(np.uint8)

    return blank_img

def bicubic_interpolation(img, scale):

    def cubic_polynomial(row, x):
        p0, p1, p2, p3 = row.astype(float)
        a = (-1/2) * p0 + (3/2) * p1 + (-3/2) * p2 + (1/2) * p3 
        b = p0 + (-5/2) * p1 + 2 * p2 + (-1/2) * p3
        c = (-1/2) * p0 + (1/2) * p2
        d = p1
        f = a * math.pow(x, 3) + b * math.pow(x, 2) + c * x + d
        for i in range(3):
            if f[i] > 255:
                f[i] = 255
            elif f[i] < 0:
                f[i] = 0
        return f

    height, width = img.shape[0], img.shape[1]
    blank_img = np.zeros((height, width, 3), np.uint8)

    for row in range(height):
        for col in range(width):
            y = row / scale
            x = col / scale
            delta_y = y - int(y)
            delta_x = x - int(x)

            if delta_y == 0 and delta_x == 0:
                blank_img[row, col] = img[int(y), int(x)]
            else:
                y_sets = [max(int(y)- 1, 0), int(y), min(int(y) + 1, int(height/scale)), min(int(y) + 2, int(height/scale))]
                x_sets = [max(int(x)- 1, 0), int(x), min(int(x) + 1, int(width/scale)), min(int(x) + 2, int(width/scale))]

                p_mat = np.array([[img[d_y, d_x] for d_x in x_sets] for d_y in y_sets])

                f_mat = np.array([cubic_polynomial(row, delta_x) for row in p_mat])

                fxy = cubic_polynomial(f_mat, delta_y)

                blank_img[row, col] = fxy.astype(np.uint8)

    return blank_img

if __name__ == "__main__":

    img = cv2.imread('test.jpg')
    height, width= img.shape[0], img.shape[1]
    h_window, w_window = math.floor(height / 3), math.floor(width / 3)

    img = exchange_position(img, h_window, w_window)

    img[-h_window:, :w_window, :] = grayscale(img[-h_window:, :w_window, :])

    img[-h_window:, -w_window:, :] = intensity_resolution(img[-h_window:, -w_window:, :])

    img[h_window:-h_window, :w_window, :] = color_filter_red(img[h_window:-h_window, :w_window, :]) 

    img[h_window:-h_window, -w_window:, :] = color_filter_yellow(img[h_window:-h_window, -w_window:, :]) 

    img[-h_window:, w_window:-w_window, :] = channel_operation(img[-h_window:, w_window:-w_window, :])

    img[:h_window, w_window:-w_window, :] = bilinear_interpolation(img[:h_window, w_window:-w_window, :], 2)

    img[h_window:-h_window, w_window:-w_window, :] = bicubic_interpolation(img[h_window:-h_window, w_window:-w_window, :], 2)

    cv2.imwrite("hw1.jpg", img)
    cv2.imshow('hw1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

