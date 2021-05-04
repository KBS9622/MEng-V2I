##############################################################################
# Data Creation
##############################################################################

import numpy as np
import matplotlib as plt


np.random.seed(2018)

N = 600

t = np.arange(0, N, 1).reshape(-1,1)
t = np.array([t[i] + np.random.rand(1)/4 for i in range(len(t))])
t = np.array([t[i] - np.random.rand(1)/7 for i in range(len(t))])
t = np.array(np.round(t, 2))

x1 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
x2 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
x3 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)

n = np.round((np.random.random(N) * 2).reshape(-1,1), 2)

y = np.array([((np.log(np.abs(2 + x1[t])) - x2[t-1]**2) + 0.02*x3[t-3]*np.exp(x1[t-1])) for t in range(len(t))])
y = np.round(y+n, 2)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,6))
ax1.set_title('Full plot')
ax1.plot(t, y, 'b-')
ax1.scatter(t, y, color='black', s=10)
ax2.set_title('Zoomed plot')
ax2.plot(t[100:200], y[100:200], 'b-')
ax2.scatter(t[100:200], y[100:200], color='black', s=10)
plt.show