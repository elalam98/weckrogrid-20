#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:49:10 2020

@author: luisaweiss
"""
import random

import numpy as np
import PIL.Image
from PIL import ImageTk, Image
from matplotlib.figure import Figure 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


# plot function is created for  
# plotting the graph in  
# tkinter window 

""" select number of random walks """
num_random_walks = 1

""" set initial seed """
if True:
    initial_seed = 13
    np.random.seed(initial_seed)
    random.seed(initial_seed)

""" random walk """
overall_rewards = []


#lithium_ion = np.array([9500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3500, 0, 0, 0, 0, 0, 0])
#solar = np.array([0, 20000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 0])    
#offshore_wind = np.array([0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
#onshore_wind = np.array([0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#pumped_hydro = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#flywheel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#vandium_redox = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3500, 0, 0, 0, 0, 0, 0, 0])
  

    
      
    # the figure that will contain the plot 
   

y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
plt.suptitle('Optimal Actions')
plt.yticks(np.arange(1,21,1))
plt.rcParams['figure.facecolor'] = 'whitesmoke'
width= 0.5
axes[0].invert_yaxis
axes[0].xaxis.set_label_position('top') 
axes[1].xaxis.set_label_position('top')
axes[0].yaxis.set_label_coords(1.14,.99)
for rw in range(num_random_walks):
    reset_environment()
    optimal_policy, rewards_path, states_path, prv_state = [], [], [], start_state
    for i in range(rl_hparams['decision_periods']):
        if rw == 0:
            if i == 2 or i == 3:
                pass

        optimal_action, _ = MGrid.q_values(prv_state)
        if optimal_action != 0:
            
            if actions_map[optimal_action][0] in power_plants.keys() and actions_map[optimal_action][1] != 0:
                    #for level in power_plants[power_plant].levels:
                nxt_state, reward_period = MGrid.compute_nextstate_reward(prv_state, optimal_action, 10)
                optimal_policy.append(actions_map[optimal_action])
               
              #  axes[1].barh(y, optimal_action, width,align='center', color= 'blue', label= optimal_policy[-1][0])
                
               # axes[1].barh(y, optimal_policy[-1][1], width,align='center', color= 'blue', label= optimal_policy[-1][0])    
            
            
            if actions_map[optimal_action][0] in storage_units.keys() and actions_map[optimal_action][1] != 0:
                        #for level in storage_units[storage_unit].levels: 
                nxt_state, reward_period = MGrid.compute_nextstate_reward(prv_state, optimal_action, 10)
                optimal_policy.append(actions_map[optimal_action])
              
               # axes[0].barh(y, optimal_action,  width,align='center',  color= 'red', label=optimal_policy[-1][0])
                
              #  axes[0].barh(y, optimal_policy[-1][1],  width,align='center',  color= 'red', label=optimal_policy[-1][0])
        rewards_path.append(reward_period)    
        states_path.append(prv_state)
        prv_state = nxt_state
    overall_rewards.append(np.mean(rewards_path))

ppquantity = [0,0,0,0,0]
suquantity = [0,0,0,0,0]
ppcost = [0,0,0,0,0]
sucost = [0,0,0,0,0]
totalqpp=0
totalqsu=0
totalcpp=0
totalcsu=0
tabledata = []
table2data = []
table3data = []

storage_color = ['lightcoral','maroon','red','lightsalmon','darkorange']
power_color = ['aqua','dodgerblue', 'blue', 'navy', 'teal']
for i in range(len(optimal_policy)):
    
    if optimal_policy[i][0] in storage_units_list:
        
        index = storage_units_list.index(optimal_policy[i][0])
        
        suquantity[index] += optimal_policy[i][1]
        
        sucost[index] += -rewards_path[i]
        
        totalqsu += optimal_policy[i][1]
        
        totalcsu += -rewards_path[i]
        
        color = storage_color[index]
        
        axes[0].barh(i+1, optimal_policy[i][1],  width,align='center',  color= color, label= optimal_policy[i][0])
        
        type = ' storage units'

    if optimal_policy[i][0] in power_plants_list:
        
        index = power_plants_list.index(optimal_policy[i][0])
        
        ppquantity[index] += optimal_policy[i][1]
        
        ppcost[index] += -rewards_path[i]
        
        totalqpp += optimal_policy[i][1]
        
        totalcpp += -rewards_path[i]
        
        color = power_color[index]

        axes[1].barh(i+1, optimal_policy[i][1], width,align='center', color= color, label= optimal_policy[i][0])
        
        type = ' power plants'
        
    tabledata.append([optimal_policy[i][0] + type, optimal_policy[i][1], -rewards_path[i]])
    
   #label= optimal_policy[i][0]

for i in range(len(suquantity)):
    table2data.append([suquantity[i], sucost[i]])
    
for i in range(len(ppquantity)):
    table3data.append([ppquantity[i], ppcost[i]])    
 
   

    
    
    
axes[0].invert_xaxis()
axes[0].yaxis.tick_right()
axes[1].yaxis.tick_left()
axes[0].tick_params(pad=15)
axes[1].tick_params(pad=15)
axes[0].set_xlabel('Storage Units (kWh)', fontsize=15)
axes[1].set_xlabel('Power Plants (kW)', fontsize=15)
axes[0].set_ylabel('Year', fontsize= 15, rotation=0, va= 'center')
fontP = FontProperties()
fontP.set_size('x-small')

handles0 = [Line2D([0], [0], color='lightcoral', lw=4, label='Lithium-ion'),Line2D([0], [0], color='maroon', lw=4, label='Lead Acid'), Line2D([0], [0], color='red', lw=4, label='Vanadium Redox'), Line2D([0], [0], color='lightsalmon', lw=4, label='Flywheel'), Line2D([0], [0], color='darkorange', lw=4, label='Pumped Hydro')]

handles1 = [Line2D([0], [0], color='aqua', lw=4, label='Solar'), Line2D([0], [0], color='dodgerblue', lw=4, label='Onshore Wind'), Line2D([0], [0], color='blue', lw=4, label='Offshore Wind'),Line2D([0], [0], color='navy', lw=4, label='Diesel')]                                                 
                                                              
axes[0].legend(handles = handles0,loc= 'upper left', bbox_to_anchor = (-0.4,1),prop=fontP)

axes[1].legend(handles = handles1,loc= 'upper right',bbox_to_anchor = (1.4,1),prop=fontP)

plt.tight_layout()

plt.show()
fig.savefig('plot.png')
#Create overall table

fig, ax = plt.subplots()

ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

columns = ('Investment Type', 'Quantity','Investment Cost ($)')

rows = ['Year %d' % x for x in range(1,21)]

results_table = plt.table(cellText = tabledata, rowLabels=rows, colLabels=columns, loc = 'center')
plt.show()
fig.savefig('table1.png')

# Create Storage Units Table
fig, ax = plt.subplots()

ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

storage_units_list2 = ['Lithium-Ion', 'Lead Acid', 'Vanadium Redox', 'Flywheel', 'Pumped Hydro', 'Total of All Storage Units']

columns = ('Total Quantity Invested', 'Total Investment Cost')

rows = storage_units_list2

table2data.append([totalqsu,totalcsu])

su_table = plt.table(cellText = table2data, rowLabels=rows, rowLoc='right', colWidths=[0.25 for x in columns], colLabels=columns, loc = 'center right')
            
plt.show()
fig.savefig('table2.png')
# Create Power Plants Table
fig, ax = plt.subplots()

ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

power_plants_list2 = ['Solar', 'Onshore Wind', 'Offshore Wind', 'Diesel', 'Hydro', ' Total of All Power Plants']

columns = ('Total Quantity Invested', 'Total Investment Cost')

rows = power_plants_list2

table3data.append([totalqpp,totalcpp])

pp_table = plt.table(cellText = table3data, rowLabels=rows, rowLoc='right', colWidths=[0.25 for x in columns], colLabels=columns, loc = 'center right')

    
plt.show()
fig.savefig('table3.png')
    #the_table = plt.table(cellText=optimal_policy,
                    #  rowLabels=decision_periods,
                   #   rowColours=colors,
                   #   loc='bottom')
   # plt.subplots_adjust(left=0.2, bottom=0.2)
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
  
      
