#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:09:13 2020

@author: elalam98
"""
import tkinter as tk
from tkinter import Tk, Label, Entry, StringVar
from tkinter import ttk
import PIL.Image
from PIL import ImageTk, Image
from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import pandas as pd
import numpy as np
# plot function is created for  
# plotting the graph in  
# tkinter window 
facility_names = ['Hospital', 'Outpatient', 'Supermarket', 'Hotel', 'Office', 'School', 'Restaurant', 'Residential']
category_names = ['lithium-ion', 'solar','offshore wind', 'onshore wind', 'pumped hydro', 'flywheel', 'vandium redox']

lithium_ion = np.array([9500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3500, 0, 0, 0, 0, 0, 0])
solar = np.array([0, 20000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 0])    
offshore_wind = np.array([0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
onshore_wind = np.array([0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pumped_hydro = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
flywheel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vandium_redox = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3500, 0, 0, 0, 0, 0, 0, 0])
  
def plot(parent): 
    
      
    # the figure that will contain the plot 
   
    fig = plt.figure(figsize= (6,3)) 
    y = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
    plt.suptitle('Optimal Actions')
    plt.yticks(np.arange(min(y), max(y)+1, 1.0))
    plt.rcParams['figure.facecolor'] = 'whitesmoke'
    width= 0.5
    axes[0].invert_yaxis
    axes[0].xaxis.set_label_position('top') 
    axes[1].xaxis.set_label_position('top')
    axes[0].yaxis.set_label_coords(1.15,1.02)
    axes[0].barh(y, lithium_ion,  width,align='center',  color= 'red', label='lithium-ion')
    axes[0].barh(y, vandium_redox, width,align='center', color= 'm', label= 'vandium redox')
    axes[1].barh(y, solar, width,align='center', color= 'blue', label= 'solar')
    axes[1].barh(y, offshore_wind,  width,align='center', color= 'y',  label= 'offshore wind')
    axes[1].barh(y, onshore_wind, width,align='center', color= 'green', label= 'onshore wind')
    axes[1].barh(y, pumped_hydro, width,align='center', color= 'orange', label= 'pumped hydro')
    axes[1].barh(y, flywheel, width,align='center', color= 'm', label= 'flywheel')
    axes[0].invert_xaxis()
    axes[0].yaxis.tick_right()
    axes[0].tick_params(pad=15)
    axes[1].tick_params(pad=15)
    axes[0].set_xlabel('Storage Units (kWh)')
    axes[1].set_xlabel('Power Plants (kW)')
    axes[0].set_ylabel('Year')
    axes[0].legend(loc= 'upper left', ncol = 1)
    axes[1].legend(loc= 'upper right', ncol= 2)
    
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, master = parent)   
    canvas.draw() 
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
        
def create_window():
    newwindow = tk.Toplevel(window)
    newwindow.geometry("800x800")
    return newwindow

#def callbackFunc(event):
#     print("New Element Selected")
def callback():
    print ("clicked!")
    print ("Their Company Name is", centry.get())
    print (hentry.get(), "Hospitals")
    print (oentry.get(), "Outpatients")
    print (sentry.get(), "Supermarkets")
    print (hotentry.get(), "Hotels")
    print (offentry.get(), "Offices")
    print (schentry.get(), "Schools")
    print (restentry.get(), "Restaurants")
    print (residentry.get(), "Residentials")

def clear_search1(event):
    centry.delete(0, tk.END)
    centry.config(fg = 'black')
def clear_search2(event):
    hentry.delete(0, tk.END)
    hentry.config(fg = 'black')
def clear_search3(event):
    oentry.delete(0, tk.END)
    oentry.config(fg = 'black')
def clear_search4(event):
    sentry.delete(0, tk.END)
    sentry.config(fg = 'black')
def clear_search5(event):
    hotentry.delete(0, tk.END)
    hotentry.config(fg = 'black')
def clear_search6(event):
    offentry.delete(0, tk.END)
    offentry.config(fg = 'black')
def clear_search7(event):
    schentry.delete(0, tk.END)
    schentry.config(fg = 'black')
def clear_search8(event):
    restentry.delete(0, tk.END)
    restentry.config(fg = 'black')
def clear_search9(event):
    residentry.delete(0, tk.END)
    residentry.config(fg = 'black')


'Hospital', 'Outpatient', 'Supermarket', 'Hotel', 'Office', 'School', 'Restaurant', 'Residential'
window = tk.Tk()
window.title('Optimal Actions for your Microgrid')
window.attributes("-fullscreen", True)
window.configure(bg = 'floral white')

frame1 = Frame(window, bg = 'floral white')
frame1.pack(side = 'top') 

frame2 = Frame(window, bg = 'floral white', pady = 50)
frame2.pack(side = 'left', expand = TRUE, anchor = NE)

frame3 = Frame(window, bg = 'floral white', pady = 50)
frame3.pack(side = 'right', expand = TRUE, anchor = NW)

frame4 = Frame(window, bg = 'floral white', pady = 150)
frame4.pack(side = 'bottom', expand = TRUE, anchor = S)

#Weckrogrid Logo
wlogo = PIL.Image.open('/Users/elalam98/Desktop/weckrogrid.png') #open image
resize = wlogo.resize((200, 65), PIL.Image.ANTIALIAS)
wlogorz = ImageTk.PhotoImage(resize)
wlogolabel = Label(frame1, image = wlogorz)
wlogolabel.pack(side = 'top', pady = 5)

# Company Name

clabel = tk.Label(frame1, text = "Company Name", fg = "Black", 
                  font = "Gotham-Thin 72", bg = 'floral white' ).pack()

centry = tk.Entry(frame1, fg = "gray", highlightbackground = 'floral white')
centry.insert(0, "Company Name")
centry.bind("<1>", clear_search1)
centry.pack(side = 'top')

# Number of Facilities

#flabel = tk.Label(frame1, 
#                  text = "Facilities:",
#                  fg = "Black",
#                  font = "Gotham-Thin 36").pack(side = 'top')
#fentry = tk.Entry(frame1, fg = "gray")
#fentry.insert(0, "Number of Facilities")
#fentry.bind("<1>", clear_search3)
#fentry.pack(side = 'top')

hlabel = tk.Label(frame2, text = "Hospitals:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
hentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
hentry.insert(0, "# of Hospitals")
hentry.bind("<1>", clear_search2)
hentry.pack(side = 'top', pady = 11)
olabel = tk.Label(frame2, text = "Outpatients:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
oentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
oentry.insert(0, "# of Outpatients")
oentry.bind("<1>", clear_search3)
oentry.pack(side = 'top', pady = 1)
slabel = tk.Label(frame2, text = "Supermarkets:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
sentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
sentry.insert(0, "# of Supermarkets")
sentry.bind("<1>", clear_search4)
sentry.pack(side = 'top', pady = 11)
hotlabel = tk.Label(frame2, text = "Hotels:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
hotentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
hotentry.insert(0, "# of Hotels")
hotentry.bind("<1>", clear_search5)
hotentry.pack(side = 'top', pady = 2)
offlabel = tk.Label(frame2, text = "Offices:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
offentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
offentry.insert(0, "# of Offices")
offentry.bind("<1>", clear_search6)
offentry.pack(side = 'top', pady = 10)
schlabel = tk.Label(frame2, text = "Schools:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
schentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
schentry.insert(0, "# of Schools")
schentry.bind("<1>", clear_search7)
schentry.pack(side = 'top', pady = 2)
restlabel = tk.Label(frame2, text = "Restaurants:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
restentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
restentry.insert(0, "# of Restaurants")
restentry.bind("<1>", clear_search8)
restentry.pack(side = 'top', pady = 9)
residlabel = tk.Label(frame2, text = "Residentials:", fg = "Black",
                  font = "Gotham-Thin 28", bg = 'floral white').pack(side = 'top')
residentry = tk.Entry(frame3, fg = "gray", highlightbackground = 'floral white')
residentry.insert(0, "# of Residentials")
residentry.bind("<1>", clear_search9)
residentry.pack(side = 'top', pady = 3)


# Type of Facilities

#tflabel = tk.Label(frame1,
#                   text = "Type of Facilities:",
#                   fg = "Black",
#                   font = "Gotham 14").pack(side = 'top')
#tfentry = tk.Entry(frame3, fg = "gray")
#tfentry.insert(0, "Type of Facilities")
#tfentry.bind("<Enter>", clear_search3)
#tfentry.pack(side = 'bottom')


#tfentry = ttk.Combobox(frame1, values=['Hospital', 'Outpatient', 
#                                       'Supermarket', 'Hotel', 
#                                       'Office', 'School', 'Restaurant', 
#                                       'Residential'])
#tfentry.pack(side = 'top')
# Plot Button

pbutton = tk.Button(frame4, text = "Plot", highlightbackground = 'floral white',
                    command = lambda:[plot(create_window()), callback()])
pbutton.pack(side = 'bottom', anchor = S)                    


# New Window Stuff

# Status Label

#status_label = Label(frame4, text = 'test', bd = 1, relief = SUNKEN, anchor = E)
#status_label.pack(fill = X, side = BOTTOM, ipady = 2)

window.mainloop()