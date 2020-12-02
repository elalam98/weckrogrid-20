#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:03:11 2020

@author: luisaweiss
"""



print('run')

import json
import collections
from collections import OrderedDict

with open('myfile.json') as f:
  data = json.load(f)
  data1 = list(data.values())[0]
  data2 = list(data.values())[1]
  data3 = list(data.values())[2]



dict1 = {}

##rename solar p[ower to solar in data2
for i in data2:
    if "Solar Power" in i['name']:
        i['name'] = 'solar'
## end rname solar power to solar in data 2
for i in data2:
    if "Onshore Wind Power" in i['name']:
        i['name'] = 'onshore wind'

for i in data2:
    if "Offshore Wind Power" in i['name']:
        i['name'] = 'offshore wind'
        
for i in data2:
    if "Diesel Power" in i['name']:
        i['name'] = 'diesel'

for i in data2:
    if "Hydro Power" in i['name']:
        i['name'] = 'hydro'

for i in data1:
    
    #dict[i[0]] = i[1]
    #print(i[0],i[1])
   # print(i['name'],i['value'])
    dict1[i['name']] = i['value']

for i in data3:
    if "Li-Ion Battery" in i['name']:
        i['name'] = 'lithium-ion'

for i in data3:
    if "Lead-Acid Battery" in i['name']:
        i['name'] = 'lead acid'

for i in data3:
    if "Vanadium Redox Battery" in i['name']:
        i['name'] = 'vanadium redox'

for i in data3:
    if "Flywheel Battery" in i['name']:
        i['name'] = 'flywheel'    

for i in data3:
    if "Pumped Hydro Battery" in i['name']:
        i['name'] = 'pumped hydro'


p = []
for i in data2:
    p.append(i['name'])

s = []             
for i in data3:
    s.append(i['name'])
    
from PlotOptimalPolicy import *
 



#for k, v in data1.items():
#    print(k, v)


# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
#print(data)
#print(data)
#print ("There inputs are : " + str(data))
#print(data1)
#print(data2)
#print(data3