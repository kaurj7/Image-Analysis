from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import time
import sys


def erosion(classNames, erosionList):

    for x in range(len(erosionList)):
        cntr = 1
        for eachImage in (erosionList[i]):
            height, width = eachImage.shape

            newErodedImage = np.zeros((height, width), dtype=np.uint8)

            k = 7
            element = np.ones((k, k), dtype=np.uint8)
            constant = (k - 1) // 2

            for i in range(constant, height - constant):
                for j in range(constant, width - constant):
                    temp = eachImage[i - constant:i + constant + 1, j - constant:j + constant + 1]
                    x = temp * element
                    newErodedImage[i, j] = np.min(x)

            cv2.imwrite(config["saveErosionImages"] + classNames[i] + str(cntr) + ".bmp", eachImage)
            cntr += 1


def dilation(classNames, dilationList):

    for x in range(len(dilationList)):
        cntr = 1
        for eachImage in (dilationList[i]):
            height, width = eachImage.shape

            newDilatedImage = np.zeros((height, width), dtype=np.uint8)

            # element = np.array([[0,1,0], [1,1,1], [0,1,0]])
            element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            constant = 1

            for i in range(constant, height - constant):
                for j in range(constant, width - constant):
                    temp = eachImage[i - constant:i + constant + 1, j - constant:j + constant + 1]
                    x = temp * element
                    newDilatedImage[i, j] = np.max(x)

            cv2.imwrite(config["saveDilationImages"] + classNames[i] + str(cntr) + ".bmp", eachImage)
            cntr += 1


def compassOperatorWithEigth(classNames, compassListEight, k1, k2, k3, k4, k5, k6, k7, k8):

    for x in range(len(compassListEight)):
        cntr = 1
        for eachImage in (compassListEight[i]):
            height, width = eachImage.shape

            newgradientImage = np.zeros((height, width))

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    k1Grad = (k1[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k1[0, 1] * eachImage[i - 1, j]) + \
                     (k1[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k1[1, 0] * eachImage[i, j - 1]) + \
                     (k1[1, 1] * eachImage[i, j]) + \
                     (k1[1, 2] * eachImage[i, j + 1]) + \
                     (k1[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k1[2, 1] * eachImage[i + 1, j]) + \
                     (k1[2, 2] * eachImage[i + 1, j + 1])

                    k2Grad = (k2[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k2[0, 1] * eachImage[i - 1, j]) + \
                     (k2[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k2[1, 0] * eachImage[i, j - 1]) + \
                     (k2[1, 1] * eachImage[i, j]) + \
                     (k2[1, 2] * eachImage[i, j + 1]) + \
                     (k2[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k2[2, 1] * eachImage[i + 1, j]) + \
                     (k2[2, 2] * eachImage[i + 1, j + 1])

                    k3Grad = (k3[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k3[0, 1] * eachImage[i - 1, j]) + \
                     (k3[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k3[1, 0] * eachImage[i, j - 1]) + \
                     (k3[1, 1] * eachImage[i, j]) + \
                     (k3[1, 2] * eachImage[i, j + 1]) + \
                     (k3[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k3[2, 1] * eachImage[i + 1, j]) + \
                     (k3[2, 2] * eachImage[i + 1, j + 1])

                    k4Grad = (k4[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k4[0, 1] * eachImage[i - 1, j]) + \
                     (k4[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k4[1, 0] * eachImage[i, j - 1]) + \
                     (k4[1, 1] * eachImage[i, j]) + \
                     (k4[1, 2] * eachImage[i, j + 1]) + \
                     (k4[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k4[2, 1] * eachImage[i + 1, j]) + \
                     (k4[2, 2] * eachImage[i + 1, j + 1])

                    k5Grad = (k5[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k5[0, 1] * eachImage[i - 1, j]) + \
                     (k5[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k5[1, 0] * eachImage[i, j - 1]) + \
                     (k5[1, 1] * eachImage[i, j]) + \
                     (k5[1, 2] * eachImage[i, j + 1]) + \
                     (k5[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k5[2, 1] * eachImage[i + 1, j]) + \
                     (k5[2, 2] * eachImage[i + 1, j + 1])

                    k6Grad = (k6[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k6[0, 1] * eachImage[i - 1, j]) + \
                     (k6[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k6[1, 0] * eachImage[i, j - 1]) + \
                     (k6[1, 1] * eachImage[i, j]) + \
                     (k6[1, 2] * eachImage[i, j + 1]) + \
                     (k6[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k6[2, 1] * eachImage[i + 1, j]) + \
                     (k6[2, 2] * eachImage[i + 1, j + 1])

                    k7Grad = (k7[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k7[0, 1] * eachImage[i - 1, j]) + \
                     (k7[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k7[1, 0] * eachImage[i, j - 1]) + \
                     (k7[1, 1] * eachImage[i, j]) + \
                     (k7[1, 2] * eachImage[i, j + 1]) + \
                     (k7[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k7[2, 1] * eachImage[i + 1, j]) + \
                     (k7[2, 2] * eachImage[i + 1, j + 1])

                    k8Grad = (k8[0, 0] * eachImage[i - 1, j - 1]) + \
                     (k8[0, 1] * eachImage[i - 1, j]) + \
                     (k8[0, 2] * eachImage[i - 1, j + 1]) + \
                     (k8[1, 0] * eachImage[i, j - 1]) + \
                     (k8[1, 1] * eachImage[i, j]) + \
                     (k8[1, 2] * eachImage[i, j + 1]) + \
                     (k8[2, 0] * eachImage[i + 1, j - 1]) + \
                     (k8[2, 1] * eachImage[i + 1, j]) + \
                     (k8[2, 2] * eachImage[i + 1, j + 1])

                    # Edge Magnitude
                    mag = np.sqrt(
                        pow(k1Grad, 2.0) + pow(k2Grad, 2.0) + pow(k3Grad, 2.0) + pow(k4Grad, 2.0) + pow(k5Grad, 2.0) + pow(
                            k6Grad, 2.0) + pow(k7Grad, 2.0) + pow(k8Grad, 2.0))
                    newgradientImage[i - 1, j - 1] = mag

            cv2.imwrite(config["saveCompass8Images"] + classNames[i] + str(cntr) + ".bmp", eachImage)
            cntr += 1


def compassOperatorWithFour(classNames, compassList, k1, k2, k3, k4):

    for x in range(len(compassList)):
        cntr = 1
        for eachImage in (compassList[i]):
            height, width = eachImage.shape

            newgradientImage = np.zeros((height, width))

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    k1Grad = (k1[0, 0] * eachImage[i - 1, j - 1]) + \
                            (k1[0, 1] * eachImage[i - 1, j]) + \
                            (k1[0, 2] * eachImage[i - 1, j + 1]) + \
                            (k1[1, 0] * eachImage[i, j - 1]) + \
                            (k1[1, 1] * eachImage[i, j]) + \
                            (k1[1, 2] * eachImage[i, j + 1]) + \
                            (k1[2, 0] * eachImage[i + 1, j - 1]) + \
                            (k1[2, 1] * eachImage[i + 1, j]) + \
                            (k1[2, 2] * eachImage[i + 1, j + 1])

                    k2Grad = (k2[0, 0] * eachImage[i - 1, j - 1]) + \
                            (k2[0, 1] * eachImage[i - 1, j]) + \
                            (k2[0, 2] * eachImage[i - 1, j + 1]) + \
                            (k2[1, 0] * eachImage[i, j - 1]) + \
                            (k2[1, 1] * eachImage[i, j]) + \
                            (k2[1, 2] * eachImage[i, j + 1]) + \
                            (k2[2, 0] * eachImage[i + 1, j - 1]) + \
                            (k2[2, 1] * eachImage[i + 1, j]) + \
                            (k2[2, 2] * eachImage[i + 1, j + 1])

                    k3Grad = (k3[0, 0] * eachImage[i - 1, j - 1]) + \
                            (k3[0, 1] * eachImage[i - 1, j]) + \
                            (k3[0, 2] * eachImage[i - 1, j + 1]) + \
                            (k3[1, 0] * eachImage[i, j - 1]) + \
                            (k3[1, 1] * eachImage[i, j]) + \
                            (k3[1, 2] * eachImage[i, j + 1]) + \
                            (k3[2, 0] * eachImage[i + 1, j - 1]) + \
                            (k3[2, 1] * eachImage[i + 1, j]) + \
                            (k3[2, 2] * eachImage[i + 1, j + 1])

                    k4Grad = (k4[0, 0] * eachImage[i - 1, j - 1]) + \
                            (k4[0, 1] * eachImage[i - 1, j]) + \
                            (k4[0, 2] * eachImage[i - 1, j + 1]) + \
                            (k4[1, 0] * eachImage[i, j - 1]) + \
                            (k4[1, 1] * eachImage[i, j]) + \
                            (k4[1, 2] * eachImage[i, j + 1]) + \
                            (k4[2, 0] * eachImage[i + 1, j - 1]) + \
                            (k4[2, 1] * eachImage[i + 1, j]) + \
                            (k4[2, 2] * eachImage[i + 1, j + 1])

                    # Edge Magnitude
                    mag = np.sqrt(pow(k1Grad, 2.0) + pow(k2Grad, 2.0) + pow(k3Grad, 2.0) + pow(k4Grad, 2.0))
                    newgradientImage[i - 1, j - 1] = mag

            cv2.imwrite(config["saveCompass4Images"] + classNames[i] + str(cntr) + ".bmp", eachImage)
            cntr += 1


def prewittAndSobel(classNames, prewittSobelList, prewittHorizontal, prewittVertical, prewittORsobel):
    for x in range(len(prewittSobelList)):
        cntr = 1
        for eachImage in (prewittSobelList[i]):
            height, width = eachImage.shape

            newgradientImage = np.zeros((height, width))

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    horizontalGrad = (prewittHorizontal[0, 0] * eachImage[i - 1, j - 1]) + \
                                     (prewittHorizontal[0, 1] * eachImage[i - 1, j]) + \
                                     (prewittHorizontal[0, 2] * eachImage[i - 1, j + 1]) + \
                                     (prewittHorizontal[1, 0] * eachImage[i, j - 1]) + \
                                     (prewittHorizontal[1, 1] * eachImage[i, j]) + \
                                     (prewittHorizontal[1, 2] * eachImage[i, j + 1]) + \
                                     (prewittHorizontal[2, 0] * eachImage[i + 1, j - 1]) + \
                                     (prewittHorizontal[2, 1] * eachImage[i + 1, j]) + \
                                     (prewittHorizontal[2, 2] * eachImage[i + 1, j + 1])

                    verticalGrad = (prewittVertical[0, 0] * eachImage[i - 1, j - 1]) + \
                                   (prewittVertical[0, 1] * eachImage[i - 1, j]) + \
                                   (prewittVertical[0, 2] * eachImage[i - 1, j + 1]) + \
                                   (prewittVertical[1, 0] * eachImage[i, j - 1]) + \
                                   (prewittVertical[1, 1] * eachImage[i, j]) + \
                                   (prewittVertical[1, 2] * eachImage[i, j + 1]) + \
                                   (prewittVertical[2, 0] * eachImage[i + 1, j - 1]) + \
                                   (prewittVertical[2, 1] * eachImage[i + 1, j]) + \
                                   (prewittVertical[2, 2] * eachImage[i + 1, j + 1])

                    # Edge Magnitude
                    mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                    newgradientImage[i - 1, j - 1] = mag

            if prewittORsobel == "prewitt":
                cv2.imwrite(config["savePrewittImages"] + classNames[i] + str(cntr) + ".bmp", eachImage)
                cntr += 1

            else if (prewittORsobel == "sobel"):
                cv2.imwrite(config["saveSobelImages"] + classNames[i] + str(cntr) + ".bmp", eachImage)
                cntr += 1


def averageFilter(classNames, averageList, kernelSize, config):

    startTime = time.time()

    index = kernelSize // 2

    for i in range(len(averageList)):
        cntr = 1
        for eachImage in (averageList[i]):
            newImage = np.copy(eachImage)

            height, width = eachImage.shape

            for x in range(height):
                for y in range(width):

                    total = 0
                    for xi in range(max(0, x - index), min(height - 1, x + index) + 1):
                        for yi in range(max(0, y - index), min(width - 1, y + index) + 1):
                            total += eachImage[xi, yi]
                    newImage[x, y] = total / (kernelSize ** 2)

            cv2.imwrite(config["averageFilterSaveLocation"] + classNames[i] + str(cntr) + ".bmp", newImage)
            cntr += 1

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def medianFilter(classNames, medianList, kernelSize, config):

    startTime = time.time()

    medianFilterList = []
    index = kernelSize // 2

    for x in range(len(medianList)):
        cntr = 1
        for eachImage in (medianList[x]):

            newImage = np.copy(eachImage)

            for i in range(len(eachImage)):
                for j in range(len(eachImage[0])):
                    for z in range(kernelSize):

                        if i + z - index < 0 or i + z - index > len(eachImage) - 1:
                            medianFilterList.append(0)

                        else:
                            if j + z - index < 0 or j + index > len(eachImage[0]) - 1:
                                medianFilterList.append(0)

                            else:
                                for k in range(kernelSize):
                                    medianFilterList.append(eachImage[(i + z - index)][(j + k - index)])

                    medianFilterList.sort()
                    newImage[i][j] = medianFilterList[len(medianFilterList) // 2]
                    medianFilterList = []

            cv2.imwrite(config["medianFilterSaveLocation"] + classNames[x] + str(cntr) + ".bmp", newImage)
            cntr += 1

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def equalizeHistogram(classNames, equalizationHist, config):

    startTime = time.time()

    for i in range(len(equalizationHist)):

        cntr = 1

        for eachImage in (equalizationHist[i]):
            height, width = eachImage.shape
            hist = np.zeros(256)

            for j in range(width):
                for k in range(height):
                    hist[eachImage[k, j]] += 1

            equal_image = np.copy(eachImage)

            height = equal_image.shape[0]
            width = equal_image.shape[1]

            x = hist.reshape(1, 256)
            y = np.array([])
            y = np.append(y, x[0, 0])
            for m in range(255):
                k = x[0, m + 1] + y[m]
                y = np.append(y, k)
            y = np.round((y / (height * width)) * 255)

            for n in range(height):
                for j in range(width):
                    k = equal_image[n, j]
                    equal_image[n, j] = y[k]

            figure = plt.figure()
            plt.title("Histogram")
            plt.xlabel("Intensity Level")
            plt.ylabel("Intensity Frequency")
            plt.xlim([0, 256])
            plt.plot(equal_image)
            plt.savefig(config["equalizedHist"] + classNames[i] + str(cntr) + ".jpg")
            plt.close(figure)

            cntr += 1

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def histogramRGBCalc(classNames, histogramRGBCalcList, config):

    startTime = time.time()

    for i in range(len(histogramRGBCalcList)):
        cntr = 1
        for eachImage in (histogramRGBCalcList[i]):
            height, width, ch = eachImage.shape
            hist = np.zeros([256, ch], np.int32)

            for j in range(0, height):
                for k in range(0, width):
                    for l in range(0, ch):
                        hist[eachImage[j, k, l], l] += 1

            figure = plt.figure()
            plt.title("color histogram")
            plt.xlabel("Intensity Level")
            plt.ylabel("Intensity Frequency")
            plt.xlim([0, 256])
            plt.plot(hist[:, 0], 'b')
            plt.plot(hist[:, 1], 'g')
            plt.plot(hist[:, 2], 'r')
            plt.savefig(config["RGBHistogram"] + classNames[i] + str(cntr) + ".jpg")
            plt.close(figure)

            cntr += 1

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def histogramAVGCalc(classNames, histogramCalcList, config):

    startTime = time.time()

    for i in range(len(histogramCalcList)):
        cntr = 1
        averageHist = np.zeros(256)

        for eachImage in (histogramCalcList[i]):
            height, width = eachImage.shape
            hist = np.zeros(256)

            for j in range(width):
                for k in range(height):
                    hist[eachImage[k, j]] += 1

            averageHist += hist
            cntr += 1

        averageHist = averageHist/len(histogramCalcList[i])
        figure2 = plt.figure()
        plt.title("Average Histogram")
        plt.xlabel("Intensity Level")
        plt.ylabel("Intensity Frequency")
        plt.xlim([0, 256])
        plt.plot(averageHist)
        plt.savefig(config["grayscaleHistogram"] + classNames[i] + "AVG.jpg")
        plt.close(figure2)

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def histogramCalc(classNames, histogramCalcList, config):

    startTime = time.time()

    for i in range(len(histogramCalcList)):
        cntr = 1
        for eachImage in (histogramCalcList[i]):
            height, width = eachImage.shape
            hist = np.zeros(256)

            for j in range(width):
                for k in range(height):
                    hist[eachImage[k, j]] += 1

            figure = plt.figure()
            plt.title("Histogram")
            plt.xlabel("Intensity Level")
            plt.ylabel("Intensity Frequency")
            plt.xlim([0, 256])
            plt.plot(hist)
            plt.savefig(config["grayscaleHistogram"] + classNames[i] + str(cntr) + ".jpg")
            plt.close(figure)

            cntr += 1

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def colorSpectrum(classNames, rgbList, color, config):

    startTime = time.time()

    for i in range(len(rgbList)):
        cntr = 1
        for eachImage in (rgbList[i]):
            b, g, r = cv2.split(eachImage)
            zeros_ch = np.zeros(eachImage.shape[0:2], dtype="uint8")

            if color == "blue":
                eachImage = cv2.merge([b, zeros_ch, zeros_ch])

            if color == "green":
                eachImage = cv2.merge([zeros_ch, g, zeros_ch])

            if color == "red":
                eachImage = cv2.merge([zeros_ch, zeros_ch, r])

            cv2.imwrite(config["colorSpectrumSaveLocation"] + classNames[i] + color + str(cntr) + ".bmp", eachImage)
            cntr += 1

    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def saltPepper(classNames, saltPepperList, strength, config):

    startTime = time.time()

    number_of_pixels = strength  # user input value
    for i in range(len(saltPepperList)):
        cntr = 1
        for eachImage in (saltPepperList[i]):
            row, col = eachImage.shape
            for j in range(number_of_pixels):
                y_coord = random.randint(0, row - 1)
                x_coord = random.randint(0, col - 1)
                eachImage[y_coord][x_coord] = 255

            for k in range(number_of_pixels):
                y_coord = random.randint(0, row - 1)
                x_coord = random.randint(0, col - 1)
                eachImage[y_coord][x_coord] = 0

            cv2.imwrite(config["saveSaltPepperImages"] + classNames[i] + str(cntr) + ".bmp", eachImage)
            cntr += 1
    endTime = time.time()
    endTime = endTime - startTime
    return endTime


def convert2Gray(grayscaleImageArray, imageType, config):
    cntr = 1
    for eachImage in range(len(grayscaleImageArray)):

        r, g, b = grayscaleImageArray[eachImage][:, :, 0], \
                  grayscaleImageArray[eachImage][:, :, 1], \
                  grayscaleImageArray[eachImage][:, :, 2]

        grayscaleImageArray[eachImage] = r * 0.2989 + g * 0.5870 + b * 0.1140
        grayscaleImageArray[eachImage] = np.uint8(grayscaleImageArray[eachImage])

        cv2.imwrite(config["saveGrayscaleImages"] + imageType + str(cntr) + ".bmp", grayscaleImageArray[eachImage])
        cntr += 1

    return grayscaleImageArray, imageType


def extractImageArrays(cellFiles, imageArrays, mypath):
    for n in range(0, len(cellFiles)):
        imageArrays[n] = cv2.imread(join(mypath, cellFiles[n]))
    return imageArrays


def readFromConfig():
    filename = "project.config"
    contents = open(filename).read()
    config = eval(contents)
    return config


def main():

    # Reading from config file for image location and saving altered image locations
    config = readFromConfig()
    mypath = config["readFileLocation"]
    print(mypath)

    # Reading user input and assigning meaningful variable names
    saltPepperStrength = int(sys.argv[1])
    colorChoice = sys.argv[2]
    medianStrength = int(sys.argv[3])
    averageStrength = int(sys.argv[4])

    # Three lists: one for each image type, second to hold grayscale images, third for rgb images
    classNames = ["cyl", "para", "inter", "super", "let", "mod", "svar"]
    classList = []
    colorClassList = []

    # For loop to iterate through each class of images
    for eachClass in classNames:
        cellFiles = [f for f in sorted(listdir(mypath))
                     if(isfile(join(mypath, f))) & (not f.startswith('.')) & (f.startswith(eachClass))]
        cellImages = np.empty(len(cellFiles), dtype=object)

        cellImageArray = extractImageArrays(cellFiles, cellImages, mypath)
        colorImageArray = np.copy(cellImageArray)
        colorClassList.append(colorImageArray)

        cellGrayArray, cylImageType = convert2Gray(cellImageArray, eachClass, config)
        cellCopyArray = np.copy(cellGrayArray)
        classList.append(cellCopyArray)

    # Copying list to avoid override changes
    saltPepperList = np.copy(classList)
    rgbList = np.copy(colorClassList)
    grayscaleHistogramList = np.copy(classList)
    avgHistList = np.copy(classList)
    rgbHistogramList = np.copy(colorClassList)
    equalizationHist = np.copy(classList)
    medianList = np.copy(classList)
    averageList = np.copy(classList)

    prewittSobelList = np.copy(classList)
    compassList = np.copy(classList)
    compassListEight = np.copy(classList)
    dilationList = np.copy(classList)
    erosionList = np.copy(classList)


    # All image processing function calls
    saltPepperTime = saltPepper(classNames, saltPepperList, saltPepperStrength, config)
    colorSpectrumTime = colorSpectrum(classNames, rgbList, colorChoice, config)
    histogramCalcTime = histogramCalc(classNames, grayscaleHistogramList, config)
    avgHistTime = histogramAVGCalc(classNames, avgHistList, config)
    histogramRGBCalcTime = histogramRGBCalc(classNames, rgbHistogramList, config)
    histEqualizationTime = equalizeHistogram(classNames, equalizationHist, config)
    medianTime = medianFilter(classNames, medianList, medianStrength, config)
    averageFilterTime = averageFilter(classNames, averageList, averageStrength, config)

    print("SaltPepper noise time: " + str(int(saltPepperTime)) + " seconds")
    print("Color Spectrum time: " + str(int(colorSpectrumTime)) + " seconds")
    print("Grayscale Histogram time: " + str(int(histogramCalcTime)) + " seconds")
    print("Average Grayscale Histogram time: " + str(int(avgHistTime)) + " seconds")
    print("RGB Histogram time: " + str(int(histogramRGBCalcTime)) + " seconds")
    print("Equalization Histogram time: " + str(int(histEqualizationTime)) + " seconds")
    print("Median filter time: " + str(int(medianTime)) + " seconds")
    print("Average filter time: " + str(int(averageFilterTime)) + " seconds")

    # Project Part2
    prewittHorizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittVertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    sobelHorizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelVertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    prewittAndSobel(classNames, prewittSobelList, prewittHorizontal, prewittVertical, "prewitt", config)
    prewittAndSobel(classNames, prewittSobelList, sobelHorizontal, sobelVertical, "sobel", config)

    k1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    k2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    k3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    k4 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])

    compassOperatorWithFour(classNames, compassList, k1, k2, k3, k4)
    compassOperatorWithEigth(classNames, compassListEight, k1, k2, k3, k4, -k1, -k2, -k3, -k4)

    dilation(classNames, dilationList)
    erosion(classNames, erosionList)


if __name__ == "__main__":
    main()
