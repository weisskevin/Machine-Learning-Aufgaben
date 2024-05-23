import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt


def linear_function_times_n(a, b, n):
    arr = np.array
    x = np.random.uniform(0,10,size=(n,1))
    y = a * x + b
    return x, y


def gaussian_noise(y, intensity):
    noise = np.random.normal(scale=intensity, size=y.shape)
    y_noisy = y + noise
    return y_noisy


def perform_classical_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def perform_robust_regression(x, y):
    model = RANSACRegressor()
    model.fit(x, y)
    return model


def add_random_outliers(x, y, n_outliers):
    x_outliers = np.random.uniform(0, 10, size=(n_outliers, 1))
    y_outliers = np.random.uniform(np.min(y), np.max(y), size=(n_outliers, 1))
    return x_outliers, y_outliers


def plot_results_gaussian(x, y, y_noisy, model_lr, model_rr, intensity):
    plt.scatter(x, y, label='Original Data', color='blue')
    plt.scatter(x, y_noisy, label=f'Noisy Data (Intensity={intensity})', color='red', alpha=0.5)
    plt.plot(x, model_lr.predict(x), color='green', linestyle='-', label='Linear Regression')
    plt.plot(x, model_rr.predict(x), color='orange', linestyle='--', label='Robust Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Linear Regression with Gaussian Noise (Intensity={intensity})')
    plt.legend()
    plt.show()


def plot_results_outliers(num_outliers, x, y, x_outliers, y_outliers, model_lr, model_rr):
    plt.scatter(x, y, label='Original Data', color='blue')
    plt.scatter(x_outliers, y_outliers, label='Outliers', color='red')
    plt.plot(x, model_lr.predict(x), color='green', linestyle='-', label='Linear Regression')
    plt.plot(x, model_rr.predict(x), color='orange', linestyle='--', label='Robust Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression with Random Outliers n='+str(num_outliers))
    plt.legend()
    plt.show()


def main():
    param_a = 3
    param_b = 5
    x_data, y_data = linear_function_times_n(param_a, param_b, 100)

    intensities = [3, 5, 8, 10, 20, 50]
    for intensity in intensities:
        y_noisy = gaussian_noise(y_data, intensity)
        model_linear_regression = perform_classical_regression(x_data, y_noisy)
        model_robust_regression = perform_robust_regression(x_data, y_noisy)
        plot_results_gaussian(x_data, y_data, y_noisy, model_linear_regression, model_robust_regression, intensity)

    num_outliers = [5, 10, 20, 30, 40]
    for num_outlier in num_outliers:
        x_data_out, y_data_out = add_random_outliers(x_data, y_data, num_outlier)
        x_data_with_outlier = np.concatenate((x_data, x_data_out))
        y_data_with_outlier = np.concatenate((y_data, y_data_out))
        model_linear_regression_outlier = perform_classical_regression(x_data_with_outlier, y_data_with_outlier)
        model_robust_regression_outlier = perform_robust_regression(x_data_with_outlier, y_data_with_outlier)
        plot_results_outliers(num_outlier, x_data, y_data, x_data_out, y_data_out, model_linear_regression_outlier, model_robust_regression_outlier)

    return 0


main()
