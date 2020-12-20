import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# camera calibration
def calibration(im, xyz, uv):
    # create matrix q
    q = np.zeros((3, 4), np.float)
    # create matrix A
    matrix = np.zeros((24, 11), np.float)
    # create matrix b
    b = np.zeros((24, 1), np.float)
    ones = np.ones((12, 1))
    # construct matrix A and matrix b
    for i in range(12):
        matrix[2 * i, :] = xyz[i][0], xyz[i][1], xyz[i][2], 1, 0, 0, 0, 0, -xyz[i][0] * uv[i][0], -xyz[i][1] * uv[i][0], -xyz[i][2] * uv[i][0]
        matrix[2 * i - 1, :] = 0, 0, 0, 0, xyz[i][0], xyz[i][1], xyz[i][2], 1, -xyz[i][0] * uv[i][1], -xyz[i][1] * uv[i][1], -xyz[i][2] * uv[i][1]
        b[2 * i, :] = uv[i][0]
        b[2 * i - 1, :] = uv[i][1]
    # calculate the 11 dofs in matrix q
    h = np.dot(np.linalg.inv(np.dot(matrix.T, matrix)), matrix.T)
    h = np.dot(h, b)
    h = list(h)
    # construct camera matrix q
    q[0, :] = h[:4]
    q[1, :] = h[4:8]
    q[2, :] = h[8], h[9], h[10], 1
    # homogenous coordinates of the world
    xyz = np.column_stack((xyz, ones))
    # projected coordinates from world coordinates
    homo_xy = np.dot(q, xyz.T)
    # homogenous coordinates of the image
    for i in range(12):
        homo_xy[0][i] = homo_xy[0][i] / homo_xy[2][i]
        homo_xy[1][i] = homo_xy[1][i] / homo_xy[2][i]
        homo_xy[2][i] = homo_xy[2][i] / homo_xy[2][i]
    # obtain the inhomogenous coordinates
    xy = homo_xy[:2, :].T

    # axis points
    axis_points = np.zeros((4, 7), np.float)
    axis_points[:, 0] = 0, 0, 0, 1
    axis_points[:, 1] = 30, 0, 0, 1
    axis_points[:, 2] = 0, 30, 0, 1
    axis_points[:, 3] = 0, 0, 30, 1
    axis_points[:, 4] = -30, 0, 0, 1
    axis_points[:, 5] = 0, -30, 0, 1
    axis_points[:, 6] = 0, 0, -30, 1
    projected_axis = np.dot(q, axis_points)
    for i in range(7):
        projected_axis[0][i] = projected_axis[0][i] / projected_axis[2][i]
        projected_axis[1][i] = projected_axis[1][i] / projected_axis[2][i]
        projected_axis[2][i] = projected_axis[2][i] / projected_axis[2][i]
    axis = projected_axis[:2, :].T

    # display visual result
    plt.imshow(im)
    mean_squared = 0
    # plot projected points and selected points
    for i in range(12):
        # green points as the selected points
        plt.plot(uv[i][0], uv[i][1], 'go-')
        # red asterisks as the projected points
        plt.plot(xy[i][0], xy[i][1], 'r*')
        mean_squared = mean_squared + np.sqrt(np.power((uv[i][0] - xy[i][0]), 2) + np.power((uv[i][1] - xy[i][1]), 2))

    # draw lines of axis
    plt.plot([axis[4][0], axis[1][0]], [axis[4][1], axis[1][1]])
    plt.plot([axis[5][0], axis[2][0]], [axis[5][1], axis[2][1]])
    plt.plot([axis[6][0], axis[3][0]], [axis[6][1], axis[3][1]])
    plt.plot(axis[0][0], axis[0][1], 'go-')
    plt.show()
    mean_squared = mean_squared / 12
    return mean_squared, q


img = plt.imread('stereo2012a.jpg')
real_coord = np.loadtxt('stereo_real.txt')
real_coord = real_coord
image_coord = np.loadtxt('stereo.txt')
error, camera = calibration(img, real_coord, image_coord)
np.savetxt('camera.txt', camera)
print(camera)
print(error)
