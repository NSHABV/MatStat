import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2, t, norm, moment
import scipy.stats as stats

gamma = 0.95
alpha = 0.05

def student_mo(samples, alpha):
    n = len(samples)
    q_1 = np.mean(samples) - np.std(samples) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    q_2 = np.mean(samples) + np.std(samples) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    return q_1, q_2

def chi_sigma(samples, alpha):
    n = len(samples)
    q_1 =  np.std(samples) * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    q_2 = np.std(samples) * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
    return q_1, q_2


def as_mo(samples, alpha):
    n = len(samples)
    q_1 = np.mean(samples) - np.std(samples) * norm.ppf(1 - alpha / 2) / np.sqrt(n)
    q_2 = np.mean(samples) + np.std(samples) * norm.ppf(1 - alpha / 2) / np.sqrt(n)
    return q_1, q_2


def as_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    U = norm.ppf(1 - alpha / 2) * np.sqrt((moment(samples, 4) / (s * s * s * s) + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2


def task8():
  samples20 = np.random.normal(0, 1, size=20)
  samples100 = np.random.normal(0, 1, size=100)
  student_20 = student_mo(samples20, alpha)
  student_100 = student_mo(samples100, alpha)
  chi_20 = chi_sigma(samples20, alpha)
  chi_100 = chi_sigma(samples100, alpha)
  as_mo_20 = as_mo(samples20, alpha)
  as_mo_100 = as_mo(samples100, alpha)
  as_d_20 = as_sigma(samples20, alpha)
  as_d_100 = as_sigma(samples100, alpha)

  print(f"Classic:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(student_100) + " \t sigma: " + str(chi_100) + "\n")

  print(f"Asymptotic:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")

  # additional task - build histogram + intervals for classic
  figure, axes = plt.subplots(2, 2, figsize = (11.8, 3.9))#three different graphics in pic
  plt.subplots_adjust(wspace=0.5)#fix title landing
  figure.suptitle("Histograms and intervals for classic", y = 1, fontsize = 20)#name graphic
  axes[0][0].hist(samples20, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][0].set_title("N(0,1) histogram,n = 20")

  axes[0][1].hist(samples100, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][1].set_title("N(0,1) histogram, n = 100")

  axes[1][0].set_ylim(-0.1, 0.5)
  axes[1][0].plot(student_20, [0,0], 'ro-', label = 'm interval, n = 20')
  axes[1][0].plot(student_100, [0.1, 0.1], 'bo-', label = 'm interval, n = 100')
  axes[1][0].legend()
  axes[1][0].set_title('m intervals')

  axes[1][1].set_ylim(-0.1, 0.5)
  axes[1][1].plot(chi_20, [0,0], 'ro-', label = 'sigma interval, n = 20')
  axes[1][1].plot(chi_100, [0.1, 0.1], 'bo-', label='sigma interval, n = 100')
  axes[1][1].legend()
  axes[1][1].set_title("sigma intervals")

  plt.show()

  # additional task - build histogram + intervals
  figure, axes = plt.subplots(2, 2, figsize = (11.8, 3.9))#three different graphics in pic
  plt.subplots_adjust(wspace=0.5)#fix title landing
  figure.suptitle("Histograms and intervals for asymptotic", y = 1, fontsize = 20)#name graphic
  axes[0][0].hist(samples20, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][0].set_title("N(0,1) histogram,n = 20")

  axes[0][1].hist(samples100, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][1].set_title("N(0,1) histogram, n = 100")

  axes[1][0].set_ylim(-0.1, 0.5)
  axes[1][0].plot(as_mo_20, [0,0], 'ro-', label = 'm interval, n = 20')
  axes[1][0].plot(as_mo_100, [0.1, 0.1], 'bo-', label = 'm interval, n = 100')
  axes[1][0].legend()
  axes[1][0].set_title('m intervals')

  axes[1][1].set_ylim(-0.1, 0.5)
  axes[1][1].plot(as_d_20, [0,0], 'ro-', label = 'sigma interval, n = 20')
  axes[1][1].plot(as_d_100, [0.1, 0.1], 'bo-', label='sigma interval, n = 100')
  axes[1][1].legend()
  axes[1][1].set_title("sigma intervals")

  plt.show()