from scipy.fft import fft, fftfreq

import numpy as np

# Number of sample points

N = 200
# sample spacing
T = 0.1/N#.0 / 400.0
fq_orig = 5
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin( 2.0*np.pi*x*fq_orig) # + 0.5*np.sin(80.0 * 2.0*np.pi*x)
y +=np.random.normal(0, 0.1, N)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt

yf_line = 2.0 / N * np.abs(yf[0:N // 2])
fq = xf[np.argmax(yf_line)]

fig,ax = plt.subplots(1,2)
ax[0].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
ax[0].axvline(fq)
ax[0].axvline(fq_orig,color='red',ls='--')
print('fq=',fq)
ax[1].plot(x, y)
ax[1].plot(x, np.sin( 2.0*np.pi*x*fq))
ax[1].plot(x, np.sin( 2.0*np.pi*x*fq_orig),color='red',ls='--')


plt.show()