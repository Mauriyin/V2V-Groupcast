import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

p = [0.1, 0.3, 0.5]
col_prob_analy = np.zeros([3, 20])
n = np.arange(10, 210, 10)
Ts = 10

for k in range(3):
    i = 0
    for density in range(10, 210, 10):
        Nv = density * 0.8
        N_ht = density * 0.1
        # P_single = (200 - density * 0.4 - 1) / (200 - density * 0.4)
        pc = 1 - (1 - p[k] / (Ts * 200))**(Nv - 1)
        col_prob_analy[k][i] = 1 - (1 - pc) * P_single**N_ht
        i = i + 1

plt.plot(n, col_prob_analy[0], '-*')
plt.plot(n, col_prob_analy[1], '-o')
plt.plot(n, col_prob_analy[2], '-d')
#plt.plot(n, simulation_p1, '-x')
plt.xlabel('Vehicle density')
plt.ylabel('Collision probability')
plt.legend(['p = 0.9', 'p = 0.7', 'p = 0.5'])
plt.show()