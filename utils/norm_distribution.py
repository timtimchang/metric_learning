import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def mix_gaussian(x):
    y1 = stats.norm.pdf(x, mu - 1 , sigma)
    y2 = stats.norm.pdf(x, mu + 1 , sigma)

    y = [ max(y1[i], y2[i]) for i in range(len(y1)) ]

    return y

def anti_gaussian(x):
    return 0


mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)


plt.plot(x,stats.norm.pdf(x, mu, sigma))
plt.show()
