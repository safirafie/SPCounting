import numpy as np
from scipy import optimize

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / (np.sqrt(2) * stddev)) ** 2)

def lorentzian(x, amplitude, mean, stddev):
    return amplitude / (1 + ((x - mean) / (np.sqrt(2) * stddev)) ** 2)

def load_data(txt_file):
    return np.loadtxt(txt_file, delimiter=',')

def fit_gaussian(hist, bins, data):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(data), np.std(data)], maxfev=10000)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
    return popt, fwhm
