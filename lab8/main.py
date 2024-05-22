import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import f_oneway

# Load the data from the file
data = np.loadtxt('0.35V_sp670.dat')

# Select the first column for analysis
data_column = data[:, 1]  # Change the column index as needed

# Step 1: Select a random section of 1024 data points
np.random.seed(0)  # For reproducibility
random_start = np.random.randint(0, len(data_column) - 1023)
selected_data = data_column[random_start:random_start + 1024]

# plt.figure(figsize=(10, 6))
# plt.plot(selected_data)
# plt.title('Исходные данные')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.hist(selected_data, bins=50)
# plt.title('Гистограмма данных')
# plt.show()

counts, bins = np.histogram(selected_data, bins=50)
background_value = bins[np.argmax(counts)]

filtered_data = medfilt(selected_data, kernel_size=5)

# plt.figure(figsize=(10, 6))
# plt.plot(filtered_data)
# plt.title('Данные после медианного фильтра')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()

# Step 6: Identify and mark different regions
# Assuming regions based on simple thresholding for illustration
signal_threshold = background_value + (bins[1] - bins[0]) * 5  # Arbitrary threshold
transition_threshold = background_value + (bins[1] - bins[0]) * 2  # Arbitrary threshold

background_region = filtered_data < transition_threshold
signal_region = filtered_data > signal_threshold
transition_region = ~background_region & ~signal_region

# plt.figure(figsize=(10, 6))
# plt.plot(filtered_data)
# plt.plot(np.where(background_region)[0], filtered_data[background_region], 'go', label='Background')
# plt.plot(np.where(signal_region)[0], filtered_data[signal_region], 'ro', label='Signal')
# plt.plot(np.where(transition_region)[0], filtered_data[transition_region], 'bo', label='Transition')
# plt.title('Данные после медианного фильтра с разметкой')
# plt.legend()
# plt.show()

background_data = filtered_data[background_region]
signal_data = filtered_data[signal_region]
transition_data = filtered_data[transition_region]
transition_data = transition_data[:-7]
k = 1 + np.log2(transition_data.__sizeof__())
i = 1

# Создание массива с группами (от 0 до 72)
groups = np.split(transition_data, 10)  # Делаем 10 точек в каждой группе (733 / 73 = 10)
variances = np.mean([np.var(group) for group in groups])
# Расчет межгрупповой дисперсии
means = [np.mean(group) for group in groups]

# Считаем дисперсию средних значений
between_group_var = np.var(means)

print("Межгрупповая дисперсия:", variances)
print("Межгрупповая дисперсия:", between_group_var)
print(between_group_var / variances)

f_val, p_val = f_oneway(background_data, signal_data, transition_data)
