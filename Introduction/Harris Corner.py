import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal


# Harris corner detection function
def harris_detection(src, sigma=2, thresh=0.01, k=0.02):
    height, width = src.shape
    # Derivative masks
    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = dx.T
    # compute x and y derivatives of image
    Ix = signal.convolve(src, dx, 'same')
    Iy = signal.convolve(src, dy, 'same')
    # generate a Gaussian kernel and perform convolution to the gradients to overcome the influence of noise
    g = cv.getGaussianKernel(ksize=13, sigma=sigma)
    Ix2 = signal.convolve(Ix * Ix, g, 'same')
    Iy2 = signal.convolve(Iy * Iy, g, 'same')
    Ixy = signal.convolve(Ix * Iy, g, 'same')
    # compute the determinant
    det =Ix2 * Iy2 - Ixy ** 2
    # compute the trace
    trace = Ix2 + Iy2
    # measure of corner response
    r = det - k * (trace ** 2)
    # get the maximum value of corner response and use this value to decide the threshold
    rmax = r.max()
    # the corner points
    points = []
    '''compare values in corner response to the threshold 
       if the value in the response is larger than the threshold
       the current point is very likely to be a corner
       perform NMS on the response'''
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # get the current 3 x 3 matrix of response
            temp = r[i -1:i + 1, j - 1:j + 1]
            # set the threshold and perform NMS
            if (r[i, j] > thresh * rmax) & (r[i, j] == temp.max()):
                # save the coordination of the corner
                points.append(tuple((i, j)))

    return points


# plot the corners on the image
def plot_harris_points(src, coords):
    height, width = src.shape
    for coord in coords:
        # change the value of corners into 255 so that these corners appear to be white
        src[coord[0], coord[1]] = 255

    return src


img1 = cv.imread('Harris_1.jpg', 0)
img2 = cv.imread('Harris_1.pgm', 0)
img3 = cv.imread('Harris_3.jpg', 0)
img4 = cv.imread('Harris_4.jpg', 0)

# cv.imshow('1', img1)
# cv.imshow('2', img2)
# cv.imshow('3', img3)
# cv.imshow('4', img4)
# cv.waitKey()

# get marked images using the above functions
points1 = harris_detection(img1)
detection1 = plot_harris_points(img1, points1)
points2 = harris_detection(img2)
detection2 = plot_harris_points(img2, points2)
points3 = harris_detection(img3)
detection3 = plot_harris_points(img3, points3)
points4 = harris_detection(img4)
detection4 = plot_harris_points(img4, points4)

# plot the images above
plt.subplot(2, 4, 1), plt.imshow(detection1, cmap='gray'), plt.title('Harris one')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.imshow(detection2, cmap='gray'), plt.title('Harris two')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 3), plt.imshow(detection3, cmap='gray'), plt.title('Harris three')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.imshow(detection4, cmap='gray'), plt.title('Harris four')
plt.xticks([]), plt.yticks([])

# get marked images using inbuilt functions
inbuilt1 = img1
inbuilt2 = img2
inbuilt3 = img3
inbuilt4 = img4
dst1 = cv.cornerHarris(inbuilt1, 2, 3, 0.02)
inbuilt1[dst1 > 0.01 * dst1.max()] = 255
dst2 = cv.cornerHarris(inbuilt2, 2, 3, 0.02)
inbuilt2[dst2 > 0.01 * dst2.max()] = 255
dst3 = cv.cornerHarris(inbuilt3, 2, 3, 0.02)
inbuilt3[dst3 > 0.01 * dst3.max()] = 255
dst4 = cv.cornerHarris(inbuilt4, 2, 3, 0.02)
inbuilt4[dst4 > 0.01 * dst4.max()] = 255

# plot these images
plt.subplot(2, 4, 5), plt.imshow(inbuilt1, cmap='gray'), plt.title('Inbuilt one')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 6), plt.imshow(inbuilt2, cmap='gray'), plt.title('Inbuilt two')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 7), plt.imshow(inbuilt3, cmap='gray'), plt.title('Inbuilt three')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 8), plt.imshow(inbuilt4, cmap='gray'), plt.title('Inbuilt four')
plt.xticks([]), plt.yticks([])
plt.show()
