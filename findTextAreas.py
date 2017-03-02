from getBox import *
import os
import cv2


DOWN, UP, LEFT, RIGHT = 1, 3, 0, 0


def splitTextArea(grid, orig, fileName):
    _, contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get contours
    for index, contour in enumerate(contours):
        [x, y, w, h] = cv2.boundingRect(contour)

        cropped = orig[y - UP : y + h + DOWN, x - LEFT: x + w + RIGHT]
        path = 'crop/' + fileName[fileName.rfind('/') + 1:fileName.rfind('.')]+'/'

        if not os.path.exists(path): os.mkdir(path)

        splitPath = path + 'crop_' + str(index) + '.jpg'

        cv2.imwrite(splitPath , cropped)


def main():
    """
    Split out the text area
    """
    extensions = {'.jpg', '.png', '.gif'}
    flowchartDataSetPath = 'flowchart_data_set/'
    dirs = os.listdir(flowchartDataSetPath)

    for imageName in dirs:
        print imageName

        if any(imageName.endswith(ext) for ext in extensions):
            img = cv2.imread(flowchartDataSetPath + imageName, cv2.IMREAD_GRAYSCALE)
            grid = wordGrid(img)

            splitTextArea(grid, img, imageName)


if __name__ == '__main__':
    main()
