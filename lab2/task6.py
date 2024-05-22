import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt

points = 20
start, end = -1.8, 2
step = 0.2
mu, sigma_squared = 0, 1
perturbations = [10, -10]  
coefs = 2, 2

def LST(x,y):
    b1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    b0 = np.mean(y) - b1 * np.mean(x)
    return b0, b1

def LMT(x,y, initial):
    fun = lambda beta: np.sum(np.abs(y - beta[0] - beta[1] * x))
    result = opt.minimize(fun, initial)
    b0 = result['x'][0]
    b1 = result['x'][1]
    return b0,b1

def find_all_coefs(x,y):
    bs0, bs1 = LST(x, y)
    bm0, bm1 = LMT(x, y, np.array([bs0, bs1]))
    return  bs0, bs1, bm0, bm1

def print_results(all_coef):
    bs0, bs1, bm0, bm1 = all_coef
    print("Критерий наименьших квадратов")
    print('a_lst = ' + str(np.around(bs0, decimals=2)))
    print('b_lst = ' + str(np.around(bs1, decimals=2)))
    print("Критерий наименьших модулей")
    print('a_lmt = ' + str(np.around(bm0, decimals=2)))
    print('b_lmt = ' + str(np.around(bm1, decimals=2)))

def criteria_comparison(x, all_coef):
    a_lst, b_lst, a_lmt, b_lmt = all_coef
    model = lambda x: coefs[0] + coefs[1] * x
    lsc = lambda x: a_lst + b_lst * x
    lmc = lambda x: a_lmt + b_lmt * x
    
    sum_lst, sum_lmt = 0, 0
    for el in x:
        y_lst = lsc(el)
        y_lmt = lmc(el)
        y_model = model(el)
        sum_lst += pow(y_model - y_lst, 2)
        sum_lmt += pow(y_model - y_lmt, 2)
        
    if sum_lst < sum_lmt:
        print("LS wins - ", sum_lst, " < ", sum_lmt) 
    else:
        print("LM wins - ", sum_lmt, " < ", sum_lst)

def plot_regression(x, y, type, estimates):
    a_ls, b_ls, a_lm, b_lm = estimates
    plt.scatter(x, y, label="Sample", edgecolor='gray', color = 'gray')
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Model', color='aqua')
    plt.plot(x, x * (b_ls * np.ones(len(x))) + a_ls * np.ones(len(x)), label='МНК', color='blue')
    plt.plot(x, x * (b_lm * np.ones(len(x))) + a_lm * np.ones(len(x)), label='МНМ', color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig(type + '.png', format='png')
    plt.show()
    plt.close()

def task6():
    print("Without pertrubations\n")
    x = np.linspace(start, end, points)
    y = coefs[0] + coefs[1] * x + stats.norm(0, 1).rvs(points)
    all_coefs = find_all_coefs(x, y)
    print_results(all_coefs)    
    criteria_comparison(x, all_coefs)
    plot_regression(x, y, "without", all_coefs)
    print("\n")
    print("With pertrubations\n")
    y[0] += perturbations[0]
    y[-1] += perturbations[1]
    all_coefs = find_all_coefs(x, y)
    print_results(all_coefs)    
    criteria_comparison(x, all_coefs)
    plot_regression(x, y, "with", all_coefs)
    print("\n")