import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import glob
import sys
import os
import sys


def projectionShadow(img, colSum, rowSum):
    h, w = np.shape(img)[:2]

    # print colSum * 1000
    # print max(colSum) / img.max()
    # print np.shape(colSum)
    # print img.max()
    imgMax = 1/(img.max())
    colSum /= imgMax
    rowSum /= imgMax

    colSumShadow = np.repeat([colSum], h, axis=0)
    rowSumShadow = np.repeat(np.array([rowSum]).T, w, axis=1)

    shadowGrid = colSumShadow + rowSumShadow

    return cv2.addWeighted(img, 0.38, shadowGrid, 0.62, 0)


def findSplicer(arr, epsilon=0.12):
    arr = covMax(arr, 50)
    threshold = epsilon * max(arr)
    # print sorted(arr), threshold

    ans = []
    start = None # if arr[0] < threshold else 1
    for i in xrange(len(arr)):
        # print arr[i]
        if arr[i] > threshold:
            if start == None:
                start = i
        elif start is not None:
            # ans.append((i + start) / 2)
            ans.append((start, i))
            start = None

    if start is not None:
        ans.append((start, len(arr)))
    # ans.append(len(arr))
    return ans

def covMax(arr, covSize):
    return [max(arr[max(0, i-covSize):min(len(arr)-1, i+covSize)]) for i in xrange(len(arr))]


def drawNicely(img):
    """
    This function is long and scary.
    But it basiclly just drawing a nice looking image.
    """

    # calculate the projections
    colSum = np.abs(np.sum(img, axis = 0))
    rowSum = np.abs(np.sum(img, axis = 1))

    # normalize the projection
    colSum /= colSum.mean() / rowSum.mean()

    colSplicer = findSplicer(colSum)
    # print colSplicer
    # for lo, hi in zip(colSplicer[:-1], colSplicer[1:]):
    for lo, hi in colSplicer:
        print lo, hi
        colImg = img[:, lo:hi]
        rowSum = np.abs(np.sum(colImg, axis = 1))
        print findSplicer(rowSum)


    colSum, rowSum = covMax(colSum, 50), covMax(rowSum, 50)

    # print colSum
    """
    In drawNicely():
    You do NOT need to know anything after this line
    """
    # build the projection shadows
    # img = projectionShadow(img, colSum, rowSum)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    bottom, height, interval, ratio = 0.1, 0.65, 0.02, np.shape(img)[1] / float(np.shape(img)[0])
    left, width = 0.1, height * ratio

    bottom_h = bottom + height + interval
    left_h = left + width + interval * ratio

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.imshow(img, cmap = 'gray')

    xymax = np.max([np.max(np.fabs(colSum)), np.max(np.fabs(rowSum))])

    axScatter.set_xlim((0, np.shape(img)[1]))
    axScatter.set_ylim((np.shape(img)[0], 0))

    axHistx.plot(xrange(len(colSum)), colSum)
    axHisty.plot(rowSum, xrange(len(rowSum)))

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()


def wordGrid(img):
    # Laplacian is a gradient based method
    # it is a good way to find edge of image, and remove the colors
    # img = abs(cv2.Laplacian(img, cv2.CV_64F, ksize = 7))

    # edges = cv2.Canny(img, 10, 20)
    # kernel = np.ones((20, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations = 1)
    # img = cv2.erode(img, kernel, iterations = 1)


    # img = cv2.GaussianBlur(img, (99, 1), 0)
    #
    # mask = np.ones_like(img)
    # mask[img == 255] = 0.

    mask = np.zeros_like(img)
    # # # print img.mean()
    mask[img > 0] = 255

    mask = 255 - mask

    # mask = (np.ones_like(mask) - mask) * 255.
    # mask[img > img.mean() * 20] = 255.0

    # kernel = np.ones((30, 1), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.dilate(mask, kernel, iterations = 1)

    kernel = np.ones((7, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)

    kernel = np.ones((7, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    kernel = np.ones((1, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    kernel = np.ones((7, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    # return mask
    # print img, mask
    mask = cv2.addWeighted(img, 0.38, mask, 0.62, 0)

    # #
    # kernel = np.ones((1, 20), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # #
    # kernel = np.ones((7, 1), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    #
    # kernel = np.ones((1, 15), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.dilate(mask, kernel, iterations = 1)


    # mask[img > 0] = 255.0
    #
    # kernel = np.ones((1, 40), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.dilate(mask, kernel, iterations = 1)

    # preserve the very small object (numbers)
    # kernel3 = np.ones((10, 3), np.uint8)
    # mask -= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)

    # mask -= largeMask

    # kernel = np.ones((1, 40), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    #
    # kernel = np.ones((1, 20), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)

    # kernel = np.ones((7, 1), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    #
    # kernel = np.ones((1, 20), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask
    plt.imshow(mask) #, cmap = 'gray')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    """
    ~ python getBox.py flowchart_data_set/flowchart1.vsd_page_1.png
    """
    fileName = sys.argv[1]
    # fileName = 'flowchart_data_set/flowchart13.vsd_page_1.png'
    # fileName = 'flowchart_data_set/flowchart91.vsd_page_1.png'
    # fileName = 'flowchart_data_set/flowchart1.vsd_page_1.png'

    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

    plt.imshow(wordGrid(img)) #, cmap = 'gray')
    plt.colorbar()
    plt.show()

    # plt.imshow(img, cmap = 'gray')
    # plt.colorbar()
    # plt.show()
