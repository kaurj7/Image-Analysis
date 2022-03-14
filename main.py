from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import time
import sys


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


if __name__ == "__main__":
    main()
