from cmath import exp, pi, polar, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from random import random

def fft_calc(y, n):

    if(n == 1):
        return [ y[0] ]

    even_y, odd_y = [], []
    for i in range(0, n, 2):
        even_y.append(y[i])
        odd_y.append(y[i+1])

    even_fft = fft_calc(even_y, n//2)
    odd_fft = fft_calc(odd_y, n//2)

    coef = []
    for k in range(n):
        coef.append( even_fft[k % (n//2)] + exp( -1j * 2*pi/n * k ) * odd_fft[k % (n//2)] )

    return coef

def ifft(coef, n):

    if(n == 1):
        return [ coef[0] ]

    even_coef, odd_coef = [], []
    for i in range(0, n, 2):
        even_coef.append(coef[i])
        odd_coef.append(coef[i+1])

    even_ifft = ifft(even_coef, n//2)
    odd_ifft = ifft(odd_coef, n//2)

    y = []
    for k in range(n):
        y.append( even_ifft[k % (n//2)] + exp( 1j * 2*pi/n * k ) * odd_ifft[k % (n//2)] )

    return y

def fft(y, n):
    result_fft = fft_calc(y, n)
    for i in range(len(result_fft)):
        result_fft[i] = complex(round(result_fft[i].real, 10), round(result_fft[i].imag, 10)) / n
    return result_fft

#USAGE

def fun1(x):
    return (np.cos(x) + np.cos(3*x) + np.cos(9*x))

def fun2(x):
    return np.random.rand(x.size)

def fun3(x):
    return 10*np.exp(-x) + fun1(x) + fun2(x)

begin = 0
end = 2*pi
n = 2 ** 7
delta = (end - begin) / n

x = np.arange(begin, end, delta)
y = fun3(x)

fft_res = fft(y.tolist(), n)
ifft_res = np.array(ifft(fft_res, n))

x = np.arange(0, n, 1)
res_freq = np.array(fft_res)

#plt.plot(x, y, 'bo')
plt.plot(x, y, "r--")
plt.plot(x, abs(res_freq), "bo")
plt.show()
