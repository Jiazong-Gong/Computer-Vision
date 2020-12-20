import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# Direct Linear Transform
def dlt(u2Trans, v2Trans, uBase, vBase):
    # create matrix A
    matrix = np.zeros((12, 9), np.float)
    for i in range(6):
        # matrix construction
        matrix[2 * i, :] = uBase[i], vBase[i], 1, 0, 0, 0, -uBase[i] * u2Trans[i], -vBase[i] * u2Trans[i], -u2Trans[i]
        matrix[2 * i + 1, :] = 0, 0, 0, uBase[i], vBase[i], 1, -uBase[i] * v2Trans[i], -vBase[i] * v2Trans[i], -v2Trans[i]
    np.savetxt('a.txt', matrix)
    # compute H using singular value decomposition
    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    v = v[-1, :]
    m = v.T
    v = np.reshape(v, (3, 3))
    v = v / v[-1, -1]
    # confirm the result is correct by checking if the multiplication of A and H is close to zero
    value = np.sum(np.dot(matrix, m))
    print(value)
    return v


# select six pixels and save them into txt file to use them again

# left = plt.imread('Left.jpg')
# plt.imshow(left)
# left_coord = plt.ginput(6)
# np.savetxt('left.txt', left_coord)
# right = plt.imread('Right.jpg')
# plt.imshow(right)
# right_coord = plt.ginput(6)
# np.savetxt('right.txt', right_coord)

# load the previously selected pixels
left_coord = np.loadtxt('left.txt')
right_coord = np.loadtxt('right.txt')

u_trans = []
v_trans = []
u_base = []
v_base = []

for i in range(6):
    left = left_coord[i]
    right = right_coord[i]
    u_trans.append(right[0])
    v_trans.append(right[1])
    u_base.append(left[0])
    v_base.append(left[1])

left = plt.imread('Left.jpg')
h = dlt(u_trans, v_trans, u_base, v_base)
# save the matrix H
np.savetxt('h.txt', h)
# m = np.loadtxt('m.txt')
building = cv.warpPerspective(left, h, dsize=(500, 370))
building = cv.cvtColor(building, cv.COLOR_BGR2RGB)
cv.imshow('', building)
cv.waitKey()
cv.imwrite('b.jpg', building)
