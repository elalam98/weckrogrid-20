#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:07:33 2020

@author: luisaweiss
"""

#!/usr/bin/python

import tkinter as tk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data1 = {'Year': ['1','2','3','4','5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
         'KiloWatts': [0, 20000, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 10]
        }
df1 = DataFrame(data1,columns=['Year','KiloWatts'])

 

root= tk.Tk() 
  
figure1 = plt.Figure(figsize=(6,5), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, root)
bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
df1 = df1[['Year','KiloWatts']].groupby('Year').sum()
df1.plot(kind='bar', legend=True, ax=ax1)
ax1.set_title('Country Vs. GDP Per Capita')


root.mainloop()