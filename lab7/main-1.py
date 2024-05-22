import numpy as np
import scipy.stats as stats
import pandas as pd


# Функция для расчета статистики хи-квадрат
def chi_square_test(observed, expected):
    return np.sum((observed - expected) ** 2 / expected)


# Параметры задачи
alpha = 0.05
n_values = [20, 100]
distributions = {
    'Normal': stats.norm,
    'Student t (k=3)': stats.t(df=3),
    'Uniform': stats.uniform
}

# Квантильные значения для уровня значимости alpha = 0.05
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
}

results = []
intervals_results = []

# Основной цикл по выборкам и распределениям
for n in n_values:
    for dist_name, dist in distributions.items():
        k = int(np.ceil(1 + 3.322 * np.log10(n)))  # Количество интервалов по формуле Стерджесса
        df = k - 1  # Степени свободы

        # Генерация выборки
        sample = dist.rvs(size=n)

        # Разбиение на интервалы
        bins = np.linspace(np.min(sample), np.max(sample), k + 1)
        observed, bin_edges = np.histogram(sample, bins)
        expected = np.diff(dist.cdf(bins)) * n

        # Вычисление статистики хи-квадрат
        chi2_value = chi_square_test(observed, expected)
        critical_value = chi2_critical_values.get(df, "N/A")
        if isinstance(critical_value, (int, float)):
            hypothesis_result = "Подтверждена" if chi2_value < critical_value else "Опровержена"
        else:
            hypothesis_result = "N/A"

        # Сохранение результатов
        results.append([dist_name, n, k, chi2_value, critical_value, critical_value - chi2_value])

        # Сохранение результатов по интервалам
        for i in range(len(observed)):
            interval = f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}"
            intervals_results.append([dist_name, n, k, interval, observed[i], expected[i], observed[i] - expected[i]])

# Вывод результатов в виде таблицы
columns = ["Распределение", "Размер Выборки", "Интервалы", "Значение Хи-квадрат", "Критическое Значение", "Результат"]
results_df = pd.DataFrame(results, columns=columns)

interval_columns = ["Распределение", "Размер Выборки", "Интервалы", "Интервал", "Наблюдаемое Значение",
                    "Ожидаемое Значение", "Разница"]
intervals_df = pd.DataFrame(intervals_results, columns=interval_columns)

# Отображение таблиц
print("Результаты проверки гипотезы хи-квадрат для различных распределений и размеров выборки:")
print(results_df.to_string(index=False))

# print("\nРезультаты по интервалам для проверки гипотезы хи-квадрат:")
# print(intervals_df.to_string(index=False))