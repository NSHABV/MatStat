import math
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats
from scipy.special import factorial
import matplotlib.pyplot as plt
import seaborn as sns
import os

#global parameters
numBins = 20
distType = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]#types of distributions

#function to get array of num of distType
def getDistribution(distType, num):
    if distType == "Normal":
        return np.random.normal(0, 1, num)
    elif distType == "Cauchy":
        return np.random.standard_cauchy(num)
    elif distType == "Laplace":
        return np.random.laplace(0, 1 / np.sqrt(2), num)
    elif distType == "Poisson":
        return np.random.poisson(10, num)
    elif distType == "Uniform":
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), num)
    return []

#function to get array of values of density function of distType
def getDensityFunc(distType, array):
    if distType == "Normal":
        return [1 / (np.sqrt(2 * np.pi)) * np.exp(-1 * x * x / 2) for x in array]
    elif distType == "Cauchy":
        return [1 / (np.pi * (x * x + 1))  for x in array]
    elif distType == "Laplace":
        return [1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(x))  for x in array]
    elif distType == "Poisson":
        return [np.power(10, x) * np.exp(-10) / factorial(x) for x in array]
    elif distType == "Uniform":
        return [1 / (2 * np.sqrt(3)) if np.fabs(x) <= np.sqrt(3) else 0  for x in array]
    return []

#task1: generate series of 10 50 1000 elements of different distributions
#Draw histogramm and graphic of density function
def task1():
    N = [10, 50, 1000]#sizes of series of data
    for distribution in distType:#for each distribution
        figure, axes = plt.subplots(1, 3, figsize = (11.8, 3.9))#three different graphics in pic
        plt.subplots_adjust(wspace=0.5)#fix title landing
        figure.suptitle(distribution + " distribution", y = 1, fontsize = 20)#name graphic
        for i in range(len(N)):
            x = getDistribution(distribution, N[i])
            n, bins, patches = axes[i].hist(x, numBins, density = 1, edgecolor = "blue", alpha = 0.3)
            axes[i].plot(bins, getDensityFunc(distribution, bins), color = "red");
            axes[i].set_title("n = " + str(N[i]))
        if not os.path.isdir("task1"):
          os.makedirs("task1")
        plt.savefig("task1/" + distribution + ".png")#save result
    plt.show()

#calculate sample mean
def calculateMean(x):
    return np.mean(x)
#calculate sample variacne
def calculateVar(x):
    return np.var(x)
#calculate sample mean
def calculateMedian(x):
    return np.median(x)
#calculate half extreme points
def calculateZR(x):
    return (min(x) + max(x)) / 2
#calculate quantile
def calculateQuantile(x, index):
    return np.quantile(x, index)
#calculate half sum quantile
def calculateZQ(x):
    return (calculateQuantile(x, 0.25) + calculateQuantile(x, 0.75)) / 2
#calculate truncated mean
def calculateZTR(x):
    r = int(len(x) * 0.25)
    sX = np.sort(x)
    sum = 0
    for i in range(r + 1, len(x) - r):
        sum += sX[i]
    return sum / (len(x) - 2 * r)

#task2: generate series of 10 100 1000 elements of different distributions
#Calculate for each:
#1)Sample mean x_
#2)Sample median - med x
#3)Half sum extreme points - ZR
#4)Half sum quantile - ZQ
#5)Truncated mean - ZTR
#repeat calculations 1000 times and calculate E(z) - expectation and D(z) variance
def task2():
    N = [10, 100, 1000]#sizes of series of data
    numOfRepeats = 1000#number of repeats

    for distribution in distType:
        result = []
        for size in N:
            mean = []
            median = []
            ZR = []
            ZQ = []
            ZTR = []
            for i in range(0, numOfRepeats):
                x = getDistribution(distribution, size)
                mean.append(calculateMean(x))
                median.append(calculateMedian(x))
                ZR.append(calculateZR(x))
                ZQ.append(calculateZQ(x))
                ZTR.append(calculateZTR(x))
            result.append(distribution + ":")
            result.append("N = " + str(size))
            result.append([" E(z) " + str(size),
                           "x_: " + str(np.around(calculateMean(mean), decimals = 6)),
                           "med: " + str(np.around(calculateMean(median), decimals = 6)),
                           "ZR: " + str(np.around(calculateMean(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateMean(ZQ), decimals = 6)),
                           "ZTR: " + str(np.around(calculateMean(ZTR), decimals = 6))])
            result.append([" D(z) " + str(size),
                           "x_: " + str(np.around(calculateVar(mean), decimals = 6)),
                           "med: " + str(np.around(calculateVar(median), decimals = 6)),
                           "ZR :" + str(np.around(calculateVar(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateVar(ZQ), decimals = 6)),
                           "ZTR :" + str(np.around(calculateVar(ZTR), decimals = 6))])
            result.append([" E(z) - sqrt(D(z)) " + str(size),
                           "x_: " + str(np.around(calculateMean(mean) - np.std(mean), decimals = 6)),
                           "med: " + str(np.around(calculateMean(median) - np.std(median), decimals = 6)),
                           "ZR :" + str(np.around(calculateMean(ZR) - np.std(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateMean(ZQ) - np.std(ZQ), decimals = 6)),
                           "ZTR :" + str(np.around(calculateMean(ZTR) - np.std(ZTR), decimals = 6))])
            result.append([" E(z) + sqrt(D(z)) " + str(size),
                           "x_: " + str(np.around(calculateMean(mean) + np.std(mean), decimals = 6)),
                           "med: " + str(np.around(calculateMean(median) + np.std(median), decimals = 6)),
                           "ZR :" + str(np.around(calculateMean(ZR) + np.std(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateMean(ZQ) + np.std(ZQ), decimals = 6)),
                           "ZTR :" + str(np.around(calculateMean(ZTR) + np.std(ZTR), decimals = 6))])
        if not os.path.isdir("task2"):
            os.makedirs("task2")
        fileName = distribution + "_data"
        completeName = os.path.join("task2/", fileName + ".txt")
        file = open(completeName, "w")
        for element in result:
            file.write(str(element) + "\n")
        file.close()

#task3: generate series of 20 100 elements of different distributions
#build for each series box plot
#generate each series 1000 times and calculate outliers
def task3():
    N = [20, 100]#sizes of series of data
    numOfRepeats = 1000#number of repeats
    #boxplot
    for distribution in distType:#for each distribution
        x20 = getDistribution(distribution, 20)
        x100 = getDistribution(distribution, 100)
        plt.boxplot((x20, x100), labels = ["n = 20", "n = 100"])
        plt.ylabel("X")
        plt.title(distribution)
        if not os.path.isdir("task3"):
          os.makedirs("task3")
        plt.savefig("task3/" + distribution + ".png")#save result
        plt.figure()
    #outliers
    result = []
    for distribution in distType:
        for size in N:
            count = 0
            for i in range(numOfRepeats):
                x = getDistribution(distribution, size)

                min = calculateQuantile(x, 0.25) - 1.5 * (calculateQuantile(x, 0.75) - calculateQuantile(x, 0.25))
                max = calculateQuantile(x, 0.75) + 1.5 * (calculateQuantile(x, 0.75) - calculateQuantile(x, 0.25))

                for j in range(size):
                    if x[j] > max or x[j] < min:
                        count += 1
            count /= numOfRepeats
            result.append(distribution + " n = " + str(size) + " number of outliers = " + str(np.around(count / size, decimals = 3)))
    if not os.path.isdir("task3"):
      os.makedirs("task3")
    completeName = os.path.join("task3/", "outliers.txt")
    file = open(completeName, "w")
    for element in result:
      file.write(str(element) + "\n")
    file.close()

def getCDF(distType, array):
    if distType == "Normal":
        return stats.norm.cdf(array)
    elif distType == "Cauchy":
        return stats.cauchy.cdf(array)
    elif distType == "Laplace":
        return stats.laplace.cdf(array)
    elif distType == "Poisson":
        return stats.poisson.cdf(array, 10)
    elif distType == "Uniform":
        return stats.uniform.cdf(array)
    return []

def getPDF(distType, array):
    if distType == "Normal":
        return stats.norm.pdf(array, 0, 1)
    elif distType == "Cauchy":
        return stats.cauchy.pdf(array)
    elif distType == "Laplace":
        return stats.laplace.pdf(array, 0, 1 / 2 ** 0.5)
    elif distType == "Poisson":
        return stats.poisson.pmf(array, 10)
    elif distType == "Uniform":
        return stats.uniform.pdf(array, -np.sqrt(3), 2 * np.sqrt(3))
    return []

def getInterval(distType):
    if distType == "Poisson":
        return (6, 14, 1)
    else:
        return (-4, 4, 0.01)

def getXs(distType):
    N = [20, 60, 100]#sizes of series of data
    result = []
    start, end, step = getInterval(distType)
    x = np.arange(start, end, step)
    for size in N:
        incorrectX = getDistribution(distType, size)
        correctX = []
        for elem in incorrectX:
            if elem >= start and elem <= end:
                correctX.append(elem)
        result.append(correctX)
    return result, x, start, end 

#task4.1: generate series of 20 60 100 elements of different distributions
#build for each series empirical distribution function
def task41():
    N = [20, 60, 100]#sizes of series of data
    for distribution in distType:
        array, x, start, end = getXs(distribution)
        index = 1
        figure, axes = plt.subplots(1, 3, figsize = (15,5))
        for elem in array:
            plt.subplot(1, 3, index)
            plt.title(distribution + ", n = " + str(N[index - 1]))
            if distribution == "Poisson" or distribution == "Uniform":
                plt.step(x, getCDF(distribution, x), color ="blue", label = "cdf")
            else:
                plt.plot(x, getCDF(distribution, x), color ="blue", label = "cdf")
            ar = np.linspace(start, end)
            ecdf = ECDF(elem)
            y = ecdf(ar)
            plt.step(ar, y, color ="black", label = "ecdf")
            plt.xlabel("x")
            plt.ylabel("(e)cdf")
            plt.legend(loc = "lower right")
            plt.subplots_adjust(wspace = 0.5)
            if not os.path.isdir("task41"):
                os.makedirs("task41")
            plt.savefig("task41/" + distribution + ".png")#save result
            index += 1

#task4.2: generate series of 20 60 100 elements of different distributions
#build for each series Kernel Density Estimation
def task42():
    N = [20, 60, 100]#sizes of series of data
    koef = [0.5, 1, 2]
    for distribution in distType:
        array, x, start, end = getXs(distribution)
        index = 1
        figure, axes = plt.subplots(1, 3, figsize = (15,5))
        for elem in array:
           headers = [r'$h = h_n/2$', r'$h = h_n$', r'$h = 2 * h_n$']
           figure, axes = plt.subplots(1, 3, figsize = (15,5))
           plt.subplots_adjust(wspace = 0.5)
           i = 0
           for k in koef:
               kde = stats.gaussian_kde(elem, bw_method = "silverman")
               hn = kde.factor
               figure.suptitle(distribution +", n =" + str(N[index - 1]))
               axes[i].plot(x, getPDF(distribution, x), color ="black", alpha = 0.5, label = "pdf")
               axes[i].set_title(headers[i])
               sns.kdeplot(elem, ax = axes[i], bw_adjust = hn * k, label= "kde", color = "blue")
               axes[i].set_xlabel('x')
               axes[i].set_ylabel('f(x)')
               axes[i].set_ylim([0, 1])
               axes[i].set_xlim([start, end])
               axes[i].legend()
               i = i + 1
               if not os.path.isdir("task42"):
                   os.makedirs("task42")
               plt.savefig("task42/" + distribution + "KDE" + str(N[index - 1]) + ".png")#save result
           index += 1


task2()