#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 03:26:13 2020

@author: elalam98
"""
import tkinter as Tk
from tkinter import * 
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import pandas as pd
import numpy as np
  
# plot function is created for  
# plotting the graph in  
# tkinter window 
def plot(): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), dpi = 100) 
    df = pd.DataFrame(np.random.randint(0,1000,size=(20, 2)), columns= ('Storage Units (kWh)', 'Power Plants (kW)'))
    y = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    fig, axes = plt.subplots(ncols=2, sharey=True)
    plt.suptitle('Optimal Actions')
    plt.yticks(np.arange(min(y), max(y)+1, 1.0))
    axes[0].invert_yaxis
    axes[0].xaxis.set_label_position('top') 
    axes[1].xaxis.set_label_position('top')
    axes[0].yaxis.set_label_coords(1.15,1.02)
    axes[0].barh(y, df['Storage Units (kWh)'], align='center', color='red')
    axes[1].barh(y, df['Power Plants (kW)'], align='center', color='blue')
    axes[0].invert_xaxis()
    axes[0].yaxis.tick_right()
    axes[0].set_xlabel('Storage Units (kWh)')
    axes[1].set_xlabel('Power Plants (kW)')
    axes[0].set_ylabel('Year')
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, master = window)   
    canvas.draw() 
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, window) 
    toolbar.update() 
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 
# the main Tkinter window 
window = Tk() 
# setting the title  
window.title('Optimal Actions for your Microgrid') 
# dimensions of the main window 
window.geometry("500x500") 
# button that displays the plot 
plot_button = Button(master = window,  
                     command = plot, 
                     height = 2,  
                     width = 10, 
                     text = "Plot") 
# place the button in main window 
plot_button.pack()  
# run the gui 
window.mainloop() 


