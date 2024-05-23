import os
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def get_voltage(filename):
  match = re.search(r'[-+]?\d*\.\d+', filename)

  if match:
    return float(match.group())
  else:
    return None

def chi_square_test(observed, expected):
    return np.sum((observed - expected) ** 2 / expected)
data_dir = 'rawData'
alpha = 0.05
chi2_critical_values = {
    1: 3.84,  # df = 1
    2: 5.99,  # df = 2
    3: 7.81,  # df = 3
    4: 9.49,  # df = 4
    5: 11.07,  # df = 5
    6: 12.59,  # df = 6
    7: 14.07,  # df = 7
    8: 15.51,  # df = 8
    9: 16.92,  # df = 9
    10: 18.31,  # df = 10
    11: 21.9200,  # df = 1
    12: 23.3367,  # df = 2
    13: 24.7356,  # df = 3
    14: 26.1189,  # df = 4
    15: 27.4884,  # df = 5
    16: 28.8454,  # df = 6
    17: 30.1910,  # df = 7
    18: 31.5264,  # df = 8
    19: 32.8523,  # df = 9
    20: 34.1696,  # df = 10
}

x_time = None
matrix_time = None
distributions = {
    'Normal': stats.norm,
    'Student t (k=3)': stats.t(df=3),
    'Uniform': stats.uniform
}
voltages = {}
results = []
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)

    if not os.path.isfile(file_path):
      continue

    voltage = get_voltage(filename)

    with open(file_path, 'r') as f:
        data = f.read().strip().splitlines()
        data_columns = [line.split(' ') for line in data]
        df = pd.DataFrame(data_columns)
        y = df.iloc[:, 1:9].values
        columns_subset_float = pd.to_numeric(y.ravel(), errors='coerce').reshape(
          y.shape).astype(float)

        if x_time is None:
            x_time = df[0].to_numpy().astype(float)

        if matrix_time is None:
            voltages[matrix_time] = y
        else:
            voltages[matrix_time] = np.vstack((matrix_time, y))

        if not voltage in voltages:
            voltages[voltage] = y.astype(float)
        else:
            voltages[voltage] = np.concatenate((voltages[voltage], y)).astype(float)

        y = voltages[voltage].flatten()
        n = x_time.__sizeof__() * 8

        k = int(np.ceil(1 + 3.322 * np.log10(n)))  # Количество интервалов по формуле Стерджесса
        a = np.min(y)
        b = np.max(y)
        bins = np.linspace(a, b, k + 1)

        # Calculate observed frequencies for the bins
        observed_freq, _ = np.histogram(y, bins=bins)

        # Calculate expected frequencies (e.g., assuming a uniform distribution)
        mu, sigma = np.mean(y), np.std(y)
        expected_freqStudent = np.diff(stats.t.cdf(bins, df=3, loc=mu, scale=sigma) * len(y)) #student
        expected_freqNormal = np.diff(stats.norm.cdf(bins, loc=mu, scale=sigma) * len(y)) #normal
        expected_freqUniform = np.full(k, len(y) / k) #uniform
        chi2_valueStudent = chi_square_test(observed_freq, expected_freqStudent)
        chi2_valueNormal = chi_square_test(observed_freq, expected_freqNormal)
        chi2_valueUniform = chi_square_test(observed_freq, expected_freqUniform)
        critical_value = chi2_critical_values.get(k - 1, "N/A")

        print("Voltage: ", voltage)
        print("valueStudent: ", chi2_valueStudent, critical_value - chi2_valueStudent)
        print("valueNormal: ", chi2_valueNormal, critical_value - chi2_valueNormal)
        print("valueUniform: ", chi2_valueUniform, critical_value - chi2_valueUniform)
        print("critValue: ", critical_value)
        print("")
        title = f'Density Histogram of Voltage Data (Number: {voltage})'

        plt.hist(y, bins=30, density=True, color='skyblue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(title)
        plt.grid(axis='y', alpha=0.75)
        plt.show()