import cv2
import numpy as np
import math
from tqdm import tqdm


def get_histogram(img):
    histogram = np.zeros(256)
    height, width = img.shape
    for row in range(height):
        for col in range(width):
            histogram[img[row, col]] += 1
    return histogram


def get_cumulative_histogram(histogram):
    cumulative_histogram = histogram
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]
    return cumulative_histogram.astype(np.int32)


def histogram_equalization(img):
    height, width = img.shape

    histogram = get_histogram(img)
    cumulative_histogram = get_cumulative_histogram(histogram)
    transform = cumulative_histogram * 255.0 / (height * width)

    new_img = img.astype(np.int32)
    for row in range(height):
        for col in range(width):
            new_img[row, col] = transform[img[row, col]]

    print(new_img)
    print(new_img.shape)
    return new_img.astype(np.uint8)


def histogram_specification(source_img, refer_img):

    height, width = source_img.shape

    source_img = histogram_equalization(source_img)
    refer_histogram = get_histogram(refer_img)
    refer_cumulative_histogram = get_cumulative_histogram(refer_histogram)

    refer_transform = (refer_cumulative_histogram * 255.0 / (refer_img.shape[0] * refer_img.shape[1])).astype(int)

    inverse_transform = np.zeros(256)
    for i in range(256):
        prev = 0
        for j in range(256):
            if refer_transform[j] >= i:
                inverse_transform[i] = prev
                break
            if refer_transform[j] != refer_transform[prev]:
                prev = j
        inverse_transform[i] = prev

    new_img = source_img.astype(np.int32)
    for row in range(height):
        for col in range(width):
            new_img[row, col] = inverse_transform[new_img[row, col]]

    return new_img.astype(np.uint8)


def gaussian_filter(img, K, kernel_size, std):

    height, width = img.shape

    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    kernel_center = (int)(kernel_size / 2)

    weighted_sum = 0
    for row in range(kernel_size):
        for col in range(kernel_size):
            r_square = (row - kernel_center) ** 2 + (col - kernel_center) ** 2
            gaussian_kernel[row, col] = K * math.exp(-r_square / (2 * (std**2)))
            weighted_sum += gaussian_kernel[row, col]

    gaussian_kernel = gaussian_kernel / weighted_sum

    new_img = np.zeros((height, width))

    for row in tqdm(range(height)):
        for col in range(width):
            sum = 0
            for s in range(kernel_size):
                for t in range(kernel_size):
                    y = row + s - 2
                    x = col + t - 2
                    if y >= 0 and y < height and x >= 0 and x < width:
                        sum += img[y, x] * gaussian_kernel[s, t]
            new_img[row, col] = sum

    print(weighted_sum)
    print(gaussian_kernel)
    print(gaussian_kernel.shape)

    return new_img.astype(np.uint8)
