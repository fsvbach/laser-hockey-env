#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:45:23 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('Plots/obsnoise.csv', delimiter=',', header=0)
data2 = pd.read_csv('Plots/obsnoise2.csv', delimiter=',', header=0)


plt.plot(data.Value ,label='high noise')
plt.plot(data2.Value ,label='low noise')
plt.legend()
plt.title('TD3 reward curve')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.savefig('Plots/td3-obsnoise.svg')
plt.show()
plt.close()