# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import scipy


def toy_recon(image):
    imh, imw = image.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    
    A = np.zeros((2 * imh * imw + 1, imh * imw))
    b = np.zeros(2 * imh * imw + 1)
    
    # Objective 1
    e = 0
    for y in range(imh):
        for x in range(imw-1):
            A[e, im2var[y, x + 1]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = image[y, x + 1] - image[y, x]
            e += 1
    
    # Objective 2
    for y in range(imh-1):
        for x in range(imw):
            A[e, im2var[y + 1, x]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = image[y + 1, x] - image[y, x]
            e += 1
            
    # Objective 3
    A[e, im2var[0, 0]] = 1
    b[e] = image[0, 0]
    
    sol = scipy.sparse.linalg.lsqr(A, b)[0]
    return sol.reshape((imh, imw))


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    imh, imw, layers = bg.shape
    maskIndices = np.argwhere(mask)
    yMin, xMin, _ = np.min(maskIndices, axis=0)
    yMax, xMax, _ = np.max(maskIndices, axis=0)
    height = yMax-yMin+1
    width = xMax-xMin+1
    fgCrop = fg[yMin:yMax+1, xMin:xMax+1, :]
    bgCrop = bg[yMin:yMax+1, xMin:xMax+1, :]
    maskCrop = mask[yMin:yMax+1, xMin:xMax+1, :]
    combinedImg = bg.copy()
    for i in range(layers):
        im2var = np.arange(height * width).reshape((height, width)).astype(int)
        
        A = np.zeros((4 * height * width + 1, height * width))
        b = np.zeros(4 * height * width + 1)
        
        # Objective 1
        e = 0
        for y in range(height):
            for x in range(width-1):
                A[e, im2var[y, x + 1]] = 1
                A[e, im2var[y, x]] = -1
                if maskCrop[y, x] and maskCrop[y, x + 1]:
                    b[e] = fgCrop[y, x + 1, i] - fgCrop[y, x, i]
                else:
                    b[e] = bgCrop[y, x + 1, i] - bgCrop[y, x, i]
                e += 1
        for y in range(height):
            for x in range(1, width):
                A[e, im2var[y, x - 1]] = 1
                A[e, im2var[y, x]] = -1
                if maskCrop[y, x] and maskCrop[y, x - 1]:
                    b[e] = fgCrop[y, x - 1, i] - fgCrop[y, x, i]
                else:
                    b[e] = bgCrop[y, x - 1, i] - bgCrop[y, x, i]
                e += 1
        
        # Objective 2
        for y in range(height-1):
            for x in range(width):
                A[e, im2var[y + 1, x]] = 1
                A[e, im2var[y, x]] = -1
                if maskCrop[y, x] and maskCrop[y + 1, x]:
                    b[e] = fgCrop[y + 1, x, i] - fgCrop[y, x, i]
                else:
                    b[e] = bgCrop[y + 1, x, i] - bgCrop[y, x, i]
                e += 1
        for y in range(1, height):
            for x in range(width):
                A[e, im2var[y - 1, x]] = 1
                A[e, im2var[y, x]] = -1
                if maskCrop[y, x] and maskCrop[y - 1, x]:
                    b[e] = fgCrop[y - 1, x, i] - fgCrop[y, x, i]
                else:
                    b[e] = bgCrop[y - 1, x, i] - bgCrop[y, x, i]
                e += 1
                
        # Objective 3
        A[e, im2var[0, 0]] = 1
        b[e] = bgCrop[0, 0, i]
        
        sol = scipy.sparse.linalg.lsqr(A, b)[0]
        combinedImg[yMin:yMax+1, xMin:xMax+1, i] = np.where(maskCrop[:, :, 0] == 1, sol.reshape((height, width)), bgCrop[:, :, i])
    return combinedImg


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    imh, imw, layers = bg.shape
    maskIndices = np.argwhere(mask)
    yMin, xMin, _ = np.min(maskIndices, axis=0)
    yMax, xMax, _ = np.max(maskIndices, axis=0)
    height = yMax-yMin+1
    width = xMax-xMin+1
    fgCrop = fg[yMin:yMax+1, xMin:xMax+1, :]
    bgCrop = bg[yMin:yMax+1, xMin:xMax+1, :]
    maskCrop = mask[yMin:yMax+1, xMin:xMax+1, :]
    combinedImg = bg.copy()
    for i in range(layers):
        im2var = np.arange(height * width).reshape((height, width)).astype(int)
        
        A = np.zeros((4 * height * width + 1, height * width))
        b = np.zeros(4 * height * width + 1)
        
        # Objective 1
        e = 0
        for y in range(height):
            for x in range(width-1):
                A[e, im2var[y, x + 1]] = 1
                A[e, im2var[y, x]] = -1
                s = fgCrop[y, x + 1, i] - fgCrop[y, x, i]
                t = bgCrop[y, x + 1, i] - bgCrop[y, x, i]
                if abs(s) >= abs(t):
                    b[e] = s
                else:
                    b[e] = t
                e += 1
        for y in range(height):
            for x in range(1, width):
                A[e, im2var[y, x - 1]] = 1
                A[e, im2var[y, x]] = -1
                s = fgCrop[y, x - 1, i] - fgCrop[y, x, i]
                t = bgCrop[y, x - 1, i] - bgCrop[y, x, i]
                if abs(s) >= abs(t):
                    b[e] = s
                else:
                    b[e] = t
                e += 1
        
        # Objective 2
        for y in range(height-1):
            for x in range(width):
                A[e, im2var[y + 1, x]] = 1
                A[e, im2var[y, x]] = -1
                s = fgCrop[y + 1, x, i] - fgCrop[y, x, i]
                t = bgCrop[y + 1, x, i] - bgCrop[y, x, i]
                if abs(s) >= abs(t):
                    b[e] = s
                else:
                    b[e] = t
                e += 1
        for y in range(1, height):
            for x in range(width):
                A[e, im2var[y - 1, x]] = 1
                A[e, im2var[y, x]] = -1
                s = fgCrop[y - 1, x, i] - fgCrop[y, x, i]
                t = bgCrop[y - 1, x, i] - bgCrop[y, x, i]
                if abs(s) >= abs(t):
                    b[e] = s
                else:
                    b[e] = t
                e += 1
                
        # Objective 3
        A[e, im2var[0, 0]] = 1
        b[e] = bgCrop[0, 0, i]
        
        sol = scipy.sparse.linalg.lsqr(A, b)[0]
        combinedImg[yMin:yMax+1, xMin:xMax+1, i] = np.where(maskCrop[:, :, 0] == 1, sol.reshape((height, width)), bgCrop[:, :, i])
    return combinedImg


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python3 main_hw2.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python3 main_hw2.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)
        
        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
