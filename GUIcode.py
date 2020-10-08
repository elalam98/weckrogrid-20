#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 03:26:13 2020

@author: elalam98
"""
import tkinter as tk
from tkinter import Tk, Label, Entry, StringVar
import PIL.Image
from PIL import ImageTk, Image
from tkinter import * 
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import pandas as pd
import numpy as np
  
# plot function is created for  
# plotting the graph in  
# tkinter window 

def plot(parent): 
    
      
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), dpi = 100) 
    df = pd.DataFrame(np.random.randint(0,1000,size=(20, 2)), columns= ('Storage Units (kWh)', 'Power Plants (kW)'))
    y = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    fig, axes = plt.subplots(ncols=2, sharey=True)
    plt.suptitle('Optimal Actions')
    plt.yticks(np.arange(min(y), max(y)+1, 1.0))
    plt.rcParams['figure.facecolor'] = 'whitesmoke'
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
    canvas = FigureCanvasTkAgg(fig, master = parent)   
    canvas.draw() 
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
        
def create_window():
    newwindow = tk.Toplevel(window)
    newwindow.geometry('500x500')
    return newwindow

    
def clear_search1(event):
    centry.delete(0, tk.END)

def clear_search2(event):
    fentry.delete(0, tk.END)

def clear_search3(event):
    tfentry.delete(0, tk.END)

window = tk.Tk()
window.title('Optimal Actions for your Microgrid')
window.geometry("500x500")

frame1 = Frame(window) 
frame1.pack(side = 'top') 

frame2 = Frame(window)
frame2.pack(side = 'left')

frame3 = Frame(window)
frame3.pack(side = 'right')

frame4 = Frame(window)
frame4.pack(side = 'bottom')

# Weckrogrid Logo
wlogo = PIL.Image.open('/Users/elalam98/Desktop/weckrogrid.png') #open image
resize = wlogo.resize((200, 65), PIL.Image.ANTIALIAS)
wlogorz = ImageTk.PhotoImage(resize)
wlogolabel = Label(frame1, image = wlogorz)
wlogolabel.pack(side = 'top', pady = 5)

# Company Name

clabel = tk.Label(frame1,
                  text = "Insert Company Name:",
                  fg = "Black",
                  font = "Gotham 16").pack()

centry = tk.Entry(frame1, fg = "gray")
centry.insert(0, "Company Name")
centry.bind("<Enter>", clear_search1)
centry.pack(side = 'top')

# Number of Facilities

flabel = tk.Label(frame2, 
                  text = "Number of Facilities:",
                  fg = "Blue",
                  font = "Gotham 14").pack(side = 'top')
fentry = tk.Entry(frame3, fg = "gray")
fentry.insert(0, "Number of Facilities")
fentry.bind("<Enter>", clear_search2)
fentry.pack(side = 'top')

# Type of Facilities

tflabel = tk.Label(frame2,
                   text = "Type of Facilities:",
                   fg = "Blue",
                   font = "Gotham 14").pack(side = 'bottom')
tfentry = tk.Entry(frame3, fg = "gray")
tfentry.insert(0, "Type of Facilities")
tfentry.bind("<Enter>", clear_search3)
tfentry.pack(side = 'bottom')

# Plot Button

pbutton = tk.Button(frame4,
                    text = "Plot",
                    command = lambda:plot(create_window()))
pbutton.pack(side = 'right')                    


# New Window Stuff

# Status Label

#status_label = Label(frame4, text = 'test', bd = 1, relief = SUNKEN, anchor = E)
#status_label.pack(fill = X, side = BOTTOM, ipady = 2)

window.mainloop()
