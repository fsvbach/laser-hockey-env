#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:45:23 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

name='stronger'

data = pd.read_csv(f'Plots/{name}.csv', delimiter=',', header=0)
data2 = pd.read_csv(f'Plots/{name}2.csv', delimiter=',', header=0)
# data3 = pd.read_csv(f'Plots/{name}3.csv', delimiter=',', header=0)


plt.plot(data.Step,data.Value ,label='td3: against strong opponent only')
plt.plot(data2.Step, data2.Value ,label='training hall: against multiple agents')
# plt.plot(data3.Step, data3.Value ,label=' ... continuation with more agents')
plt.legend()
plt.title('TD3 default rewards')
plt.xlabel('time steps')
plt.ylabel('average reward')
plt.savefig(f'Plots/td3-{name}.svg')
plt.show()
plt.close()



name='reward'

data = pd.read_csv(f'Plots/{name}.csv', delimiter=',', header=0)
data2 = pd.read_csv(f'Plots/{name}2.csv', delimiter=',', header=0)
data3 = pd.read_csv(f'Plots/{name}3.csv', delimiter=',', header=0)


plt.plot(data.Step,data.Value ,label='overfit: against strong opponent only')
plt.plot(data2.Step, data2.Value ,label='superagent: training hall with td3')
plt.plot(data3.Step, data3.Value ,label='... continuation with more agents')
plt.legend()
plt.title('TD3 new rewards')
plt.xlabel('time steps')
plt.ylabel('average reward')
plt.savefig(f'Plots/td3-{name}.svg')
plt.show()
plt.close()


name='obsnoise'

data = pd.read_csv(f'Plots/{name}.csv', delimiter=',', header=0)
data2 = pd.read_csv(f'Plots/{name}2.csv', delimiter=',', header=0)
# data3 = pd.read_csv(f'Plots/{name}3.csv', delimiter=',', header=0)


plt.plot(data.Step,data.Value ,label='high noise')
plt.plot(data2.Step, data2.Value ,label='low noise')
# plt.plot(data3.Step, data3.Value ,label=' ... continuation with more agents')
plt.legend()
plt.title('TD3 with observation noise')
plt.xlabel('time steps')
plt.ylabel('average reward')
plt.savefig(f'Plots/td3-{name}.svg')
plt.show()
plt.close()