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
from scipy.sparse.linalg import lsqr


def toy_recon(image):
    imh, imw= image.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    
    A = np.zeros((imh*imw*2,imh*imw))
    b = np.zeros((imh*imw*2,))
    e = 0 # equation counter

    for x in range(imw): 
        for y in range(imh):
            # objective 1: x grads
            if x<imw-1:
                A[e, im2var[y, x + 1]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = image[y, x + 1] - image[y, x]
                e+=1 # new row for each equation

            # objective 2: y grads
            if y<imh-1: 
                A[e, im2var[y+1, x]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = image[y+1, x] - image[y, x]
                e+=1

    # objective 3: top corner matches
    A[e, im2var[0, 0]] = 1
    b[e] = image[0, 0]
    
    A = A[:e+1,:]
    b = b[:e+1]
    v = lsqr(A,b)[0] 
    return v.reshape((imh,imw))


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    H,W,C = fg.shape
    out = bg

    # Step 1: Select source and target regions. 
    # Select the boundaries of a region in the source image and specify a location in the target image where it should be blended. 
    # Then, transform (e.g., translate) the source image so that indices of pixels in the source and target regions correspond. 
    # We’ve provided starter code that will help with this. You may want to augment the code to allow rotation or resizing into the target region. 
    # You can be a bit sloppy about selecting the source region – just make sure that the entire object is contained. 
    # Ideally, the background of the object in the source region and the surrounding area of the target region will be of similar color.
    
    bounds = np.where((mask==1) & ((mask != np.roll(mask,1,axis = 0)) | 
                                (mask != np.roll(mask,1,axis = 1)) |
                                (mask != np.roll(mask,-1,axis = 0)) | 
                                (mask != np.roll(mask,-1,axis = 1))))[:2]
    xmin,xmax = np.min(bounds[1])-1, np.max(bounds[1])+1
    ymin,ymax = np.min(bounds[0])-1, np.max(bounds[0])+1
    w,h = xmax - xmin, ymax - ymin
    
    im2var = np.arange(h*w).reshape((h,w)).astype(int)
    newmask = mask[ymin:ymax,xmin:xmax,0]
    # boundary = im2var[np.where((newmask==1) & ((newmask != np.roll(newmask,1,axis = 0)) | 
    #                             (newmask != np.roll(newmask,1,axis = 1)) |
    #                             (newmask != np.roll(newmask,-1,axis = 0)) | 
    #                             (newmask != np.roll(newmask,-1,axis = 1))))[:2]]
    # print(xmin,xmax,ymin,ymax, bounds[0].shape,newmask.shape)
    # for i in boundary:
    #     x,y = (int) (i%w), (int) (i//w)
    #     out[ymin+y,xmin+x] = np.array([1,0,0]) 
    # plt.imshow(out)
    # plt.savefig("boundary.png")
    # have the boundary be part of the mask that is on the edge, assuming mask is 1/0

    for c in range(C):
        # Step 2: Solve the blending constraints.
        # v=argminv∑i∈S,j∈Ni∩S((vi−vj)−(si−sj))2+∑i∈S,j∈Ni∩¬S((vi−tj)−(si−sj))2
        # basically: minimize gradients on the respective fg/bg based on the mask
        A = np.zeros(((h*w*3),h*w),dtype = np.float32)
        b = np.zeros(((h*w*3)),dtype = np.float32)
        e = 0
        # print(A.shape,fg.shape,mask.shape,bg.shape)
        # we are solving the following equation: (Ax - b)^2
        # for bound in boundary:
        #     A[e,bound] = 1
        #     b[e] = bg[(int)(bound//w) , (int)(bound%w),c] # want boundaries to match background
        #     e+=1
        # image = fg[:,:,c]
        # for x in range(W):
        #     for y in range(H):
        fimg = fg[ymin:ymax,xmin:xmax,c]
        bimg = bg[ymin:ymax,xmin:xmax,c]
        for x in range(w):
            for y in range(h):
                if newmask[y,x]==1:
                    image = fimg
                else: 
                    image = bimg
                    A[e,im2var[y, x]] =1
                    b[e] = bimg[y,x]
                    e+=1
                # image = bg[:,:,c] if mask[y,x]==1 else fg[:,:,c]
                if x<w-1:
                    A[e, im2var[y, x + 1]] = 1
                    A[e, im2var[y, x]] = -1
                    b[e] = image[y, x + 1] - image[y, x]
                    e+=1 # new row for each equation

                # objective 2: y grads
                if y<h-1: 
                    A[e, im2var[y+1, x]] = 1
                    A[e, im2var[y, x]] = -1
                    b[e] = image[y+1, x] - image[y, x]
                    e+=1
        # A[e,0] =1
        # b[e] = bg[0,0,c]
        # Step 3: Copy the solved values v_i into your target image. 
        A = A[:e,:]
        b = b[:e]
        v = lsqr(A,b)[0] # keeps getting killed on local -- will try on afs
        # v = np.linalg.lstsq(A, b)[0]
        # print(A.shape,v.shape,H,W)
        # out[ymin:ymax+1,xmin:xmax+1,c] = v.reshape((h,w))
        v = v.reshape((h,w))
        # for x in range(w):
        #     for y in range(h):
        #         if newmask[y,x] == 1:
        #             out[ymin+y,xmin+x,c] = v[y,x]
        out[ymin:ymax,xmin:xmax,c] = v
        out[:,:,c] = (out[:,:,c]-np.min(out[:,:,c]))/(np.max(out[:,:,c])-np.min(out[:,:,c]))
        #scale to help with the extreme values
        plt.imshow(out[:,:,c])
        plt.savefig("testing"+str(c)+".png")
        # For RGB images, process each channel separately. 
        # Show at least three results of Poisson blending. 
        # Explain any failure cases (e.g., weird colors, blurred boundaries, etc.).

        # Tips
        # For your first blending example, try something that you know should work, such as the included penguins on top of the snow in the hiking image.
        # Object region selection can be done very crudely, with lots of room around the object.
    return out


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    return fg * mask + bg * (1 - mask)


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

    # Example script: python proj2_starter.py -q toy
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

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
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

        # plt.subplot(211)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.show()
        plt.savefig(args.target.split(".")[0] + "_naive.png")
        # plt.subplot(212)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()
        plt.savefig(args.target.split(".")[0] + "_output.png")

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
