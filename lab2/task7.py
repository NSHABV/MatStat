import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
from tabulate import tabulate
from scipy.stats import laplace, uniform
import math

points = 20
sample_size_normal = 100
sample_size_laplace = 20
start, end = -1.4, 1.4
alpha = 0.05
p_ = 1 - alpha
mu_, sigma_squared = 0, 1

def find_k(size):
    return math.ceil(1.72 * (size) ** (1/3))

def max_likelihood_estimation(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    print("mu = ", np.around(mu, decimals=3),
          " sigma=", np.around(sigma, decimals=3))
    return mu, sigma

def calculate_chi2(p, n, sample_size):
    tmp = np.multiply((n - sample_size * p), (n - sample_size * p))
    chi2 = np.divide(tmp, p * sample_size)
    return chi2

def is_hypo_accepted(quantile, chi2):
    if quantile > np.sum(chi2):
        return True
    return False

def find_all_probabilities(borders, hypothesis, sample, k):
    p = np.array(hypothesis(start))
    n = np.array(len(sample[sample < start]))

    for i in range(k - 2):
        p_i = hypothesis(borders[i + 1]) - hypothesis(borders[i])
        p = np.append(p, p_i)
        n_i = len(sample[(sample < borders[i + 1]) & (sample >= borders[i])])
        n = np.append(n, n_i)

    p = np.append(p, 1 - hypothesis(end))
    n = np.append(n, len(sample[sample >= end]))
    
    return p,n

    
def chi_square_criterion(sample, mu, sigma, k):
    hypothesis = lambda x: stats.norm.cdf(x, loc = mu, scale = sigma)
    borders = np.linspace(start, end, num = k - 1)
    p, n = find_all_probabilities(borders, hypothesis, sample, k)
    chi2 = calculate_chi2(p, n, len(sample))
    quantile = stats.chi2.ppf(p_, k - 1)
    isAccepted = is_hypo_accepted(quantile, chi2)
    return chi2, isAccepted, borders, p, n

def build_table(chi2, borders, p, n, sample_size):
    rows = []
    headers = ["$i$", "$\\Delta_i = [a_{i-1}, a_i)$", "$n_i$", "$p_i$",
               "$np_i$", "$n_i - np_i$", "$(n_i - np_i)^2/np_i$"]   
    for i in range(0, len(n)):
        if i == 0:
            limits = ["$-\infty$", np.around(borders[0], decimals=3)]
        elif i == len(n) - 1:
            limits = [np.around(borders[-1], decimals=3), "$\infty$"]
        else:
            limits = [np.around(borders[i - 1], decimals=3), np.around(borders[i], decimals=3)]
        rows.append([i + 1, limits, n[i],
             np.around(p[i], decimals=4),
             np.around(p[i] * sample_size, decimals=3),
             np.around(n[i] - sample_size * p[i], decimals=3),
             np.around(chi2[i], decimals=3)] )
    rows.append(["\\sum", "--", np.sum(n), np.around(np.sum(p), decimals=4),
                 np.around(np.sum(p * sample_size), decimals=3),
                 -np.around(np.sum(n - sample_size * p), decimals=3),
                 np.around(np.sum(chi2), decimals=3)]
    )
    return tabulate(rows, headers)

def check_acception(isAccepted):
    if isAccepted:
        print("\nГипотезу принимаем")
    else:
        print("\nГипотезу принимаем!")

def calcucate_normal():
    k = find_k(sample_size_normal)
    normal_sample = np.random.normal(0, 1 , sample_size_normal)
    mu, sigma = max_likelihood_estimation(normal_sample)
    chi2, isAccepted, borders, p, n = chi_square_criterion(normal_sample, mu, sigma, k)
    print(build_table(chi2, borders, p, n, 100))
    check_acception(isAccepted)

def calcucate_laplace():
    k = find_k(sample_size_laplace)
    laplace_sample = distribution = laplace.rvs(size=20, scale=1 / math.sqrt(2), loc=0)
    mu, sigma = max_likelihood_estimation(laplace_sample)
    chi2, isAccepted, borders, p, n = chi_square_criterion(laplace_sample, mu, sigma, k)
    print(build_table(chi2, borders, p, n, 20))
    check_acception(isAccepted)

def task7():
    calcucate_normal()
    calcucate_laplace()
