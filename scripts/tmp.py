import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data
x = np.linspace(0, 10, 100)
y = np.piecewise(x, [x < 5, x >= 5], [lambda x: 2*x + 1, lambda x: -x + 20])
y += np.random.normal(scale=1.0, size=x.shape)  # add noise

# Define piecewise function
def piecewise_func(x, x0,  b1,  b2):
    return np.where(x < x0,  b1,  b2)

# Fit
popt, _ = curve_fit(piecewise_func, x, y, p0=[5,  1,  20])

# Plot
plt.scatter(x, y, label='Data')
plt.plot(x, piecewise_func(x, *popt), color='red', label='Piecewise Fit')
plt.legend()
plt.show()