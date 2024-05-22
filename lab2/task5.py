import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tabulate
from matplotlib.patches import Ellipse
import statistics
import matplotlib.transforms as transforms

sizes = [20, 60, 100]
correlation_coef = [0, 0.5, 0.9]
mean = [0, 0]
num = 1000

get_usual = lambda num, cov: stats.multivariate_normal.rvs(mean, cov, num)
get_mixed = lambda num, cov: 0.9 * stats.multivariate_normal.rvs(mean, [[1, 0.9], [0.9, 1]], num) \
             + 0.1 * stats.multivariate_normal.rvs(mean, [[10, -0.9], [-0.9, 10]], num)

def create_table(num, cov, fun):
    coefs = calculate_coefs(num, cov, fun)
    rows =[]
    rows.append(['$E(z)$', np.around(median(coefs['pearson']), decimals = 3),
                np.around(median(coefs['spearman']), decimals = 3),
                np.around(median(coefs['quadrat']), decimals = 3)])
    rows.append(['$E(z^2)$', np.around(quadric_median(coefs['pearson'], num), decimals = 3),
                np.around(quadric_median(coefs['spearman'], num), decimals = 3),
                np.around(quadric_median(coefs['quadrat'], num), decimals = 3)])
    rows.append(['$D(z)$', np.around(variance(coefs['pearson']), decimals = 3),
                np.around(variance(coefs['spearman']), decimals = 3),
                np.around(variance(coefs['quadrat']), decimals = 3)])
    return rows

def median(data):
    return np.median(data)

def quadric_median(data, num):
    return np.median([pow(data[k], 2) for k in range(num)])

def variance(data):
    return statistics.variance(data)

def calculate_coefs(num, cov, fun):
    coefs = {'pearson' : [], 'spearman' : [], 'quadrat' : []}
    for i in range(num):
        data = fun(num, cov)
        x, y = data[:, 0], data[:, 1]
        coefs['pearson'].append(stats.pearsonr(x, y)[0])
        coefs['spearman'].append(stats.spearmanr(x, y)[0])
        coefs['quadrat'].append(np.mean(np.sign(x - median(x)) * np.sign(y - median(y))))
    return coefs

def calculate_usual():
    for size in sizes:
        print("n = ", size)
        table = []
        for coef in correlation_coef:
            cov = [[1.0, coef], [coef, 1.0]]
            extension_table = create_table(size, cov, get_usual)
            title_row = ["$\\correlation_coef$ = " + str(coef), '$r$', '$r_S$', '$r_Q$']
            table.append([])
            table.append(title_row)            
            table.extend(extension_table)
        print(tabulate.tabulate(table, headers=[]))

def calculate_mixed():
    table = []
    for size in sizes:
        extension_table = create_table(size, None, get_mixed)
        title_row = ["$n = " + str(size) + "$", '$r$', '$r_S$', '$r_Q$']
        table.append(title_row)
        table.extend(extension_table)
    print(tabulate.tabulate(table, headers=[]))

def create_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    r_x = np.sqrt(1 + pearson)
    r_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=r_x * 2, height=r_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_ellipses(samples):
    plt.rcParams['figure.figsize'] = [18, 12]
    num = len(samples[0])
    fig, ax = plt.subplots(1, len(samples))
    fig.suptitle("n = " + str(num))    
    titles = ['$\\rho = 0$', '$\\rho = 0.5$', '$\\rho = 0.9$']
    i = 0
    for sample in samples:
        x = sample[:, 0]
        y = sample[:, 1]
        ax[i].scatter(x, y, c='red', s=6)
        create_ellipse(x, y, ax[i], edgecolor='gray')
        ax[i].scatter(np.mean(x), np.mean(y), c='blue', s=6)
        ax[i].set_title(titles[i])
        i += 1
    plt.savefig(
        "Ellipse n = " + str(num) + ".png",
        format='png'
    )

def ellipse_task():
    samples = []
    for num in sizes:
        for coef in correlation_coef:
            samples.append(get_usual(num, [[1.0, coef], [coef, 1.0]]))
        plot_ellipses(samples)
        samples = []

def task5():
    calculate_usual()
    calculate_mixed()
    ellipse_task()