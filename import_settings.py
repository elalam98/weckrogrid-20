"""
Created on Mon Mar 16 16:18:07 2020
Import settings
@author: Stamatis
"""

""" import libraries """
import copy
import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
import time
import sys
from collections import Counter, deque, OrderedDict
from keras import initializers
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM, GRU, Flatten, RepeatVector, TimeDistributed
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from tqdm import tqdm

""" file and datetime settings """
settings = dict()
file_path = '/Files/Rutgers/Research/Reinforcement Learning/Stam/Journal Paper 2/Code'
settings['filename_electricity'] = file_path + '/data_load_'
settings['filename_weather'] = file_path + '/data_meteorological'
settings['weather_predictors'] = ['Temperature', 'GHI', 'Wind Speed']
settings['start_date'] = '2002-01-01 01:00:00'
settings['end_date'] = '2002-12-31 23:00:00'
settings['random_transition_probabilities'] = False
settings['use_exponential_voll'] = False
settings['exponential_multiplier'] = 0.2
settings['include_ev_load'] = False
settings['superposed_scenario'] = True
settings['annual_demand_growth_rate_percent'] = 1
settings['num_outage_simulation_runs'] = 3
settings['increasing_storage_prices'] = []

""" constants """
DAYS_IN_YEAR = 365
HRS_IN_YEAR = 8760
SEC_IN_MIN = 60
SEC_IN_HOUR = 3600
ELECTRICITY_PRICE_PER_KWH = 0.1386
DIESEL_PRICE_PER_GALLON = 2.459
DIESEL_KWH_PER_GALLON = 35.8 * 3.785 * (1 / 3.6)  # (MJ/l) * (l/gal) * (KWh/MJ)
DIESEL_PRICE_PER_KWH = DIESEL_PRICE_PER_GALLON / DIESEL_KWH_PER_GALLON

""" initialization """
decision_periods, years_in_period, mc_transition_prob = 20, 1, 0.8
interest_rate, payments_per_year, loan_horizon = 0.02, 12, 10

""" define facilities, power plants and storage units lists """
facilities_list = ['hospital', 'outpatient', 'supermarket', 'hotel', 'office', 'school', 'restaurant', 'residential']
power_plants_list = ['solar', 'onshore wind', 'offshore wind', 'diesel']
storage_units_list = ['lithium-ion', 'lead acid', 'vanadium redox', 'flywheel', 'pumped hydro']

""" class for microgrid facilities """
class MGridFacility:
    def __init__(self,cnt,crit_load,voll,ev_profile,num_ev,lolp_constraint,ccp_constraint,kkt_multiplier):
        self.cnt = cnt
        self.crit_load = crit_load
        self.voll = voll
        self.ev_profile = ev_profile
        self.num_ev = num_ev
        self.lolp_constraint = lolp_constraint
        self.ccp_constraint = ccp_constraint
        self.kkt_multiplier = kkt_multiplier
        self.yearly_data = None
        self.total_data = None

""" class for power plants """
class MGridPowerPlant:
    def __init__(self,price_list,price_probs,price_state,life_list,life_probs,life_state,eff_list,eff_probs,eff_state,levels,decom_cost,annual_om_cost_rate,cap,loan_years,om_cost,rem_life,cur_eff):
        self.price_list = price_list
        self.price_probs = price_probs
        self.price_state = price_state
        self.life_list = life_list
        self.life_probs = life_probs
        self.life_state = life_state
        self.eff_list = eff_list
        self.eff_probs = eff_probs
        self.eff_state = eff_state
        self.levels = levels
        self.decom_cost = decom_cost
        self.annual_om_cost_rate = annual_om_cost_rate
        self.cap = cap
        self.loan_years = loan_years
        self.loan_payment = 0
        self.rem_loan = 0
        self.om_cost = om_cost
        self.rem_life = rem_life
        self.cur_eff = cur_eff
        self.prod = None
        self.power = None

    def install_new(self, new_cap):
        self.cap = new_cap
        self.loan_payment = - payments_per_year * np.pmt(interest_rate / payments_per_year, self.loan_years * payments_per_year, self.price_list[self.price_state]) * new_cap
        self.rem_loan = self.loan_payment * self.loan_years
        self.om_cost = years_in_period * self.annual_om_cost_rate * self.price_list[self.price_state] * new_cap
        self.rem_life = self.life_list[self.life_state] if new_cap != 0 else 0
        self.cur_eff = self.eff_list[self.eff_state] if new_cap != 0 else 0

    def decrease_life(self):
        if self.rem_life <= years_in_period:
            self.cap = 0
            self.loan_payment = 0
            self.rem_loan = 0
        self.rem_life = max(self.rem_life-years_in_period, 0)

    def state_transition(self, feature):
        if random.uniform(0, 1) < getattr(self, feature + '_probs')[getattr(self, feature + '_state')]:
            setattr(self, feature + '_state', getattr(self, feature + '_state') + 1)

""" class for storage units """
class MGridStorageUnit:
    def __init__(self,price_list,price_probs,price_state,life_list,life_probs,life_state,eff_list,eff_probs,eff_state,dod_list,dod_probs,dod_state,initial_soc,levels,decom_cost,annual_om_cost_rate,annual_degr_rate,cap,loan_years,om_cost,rem_life,cur_eff,cur_dod):
        self.price_list = price_list
        self.price_probs = price_probs
        self.price_state = price_state
        self.life_list = life_list
        self.life_probs = life_probs
        self.life_state = life_state
        self.eff_list = eff_list
        self.eff_probs = eff_probs
        self.eff_state = eff_state
        self.dod_list = dod_list
        self.dod_probs = dod_probs
        self.dod_state = dod_state
        self.initial_soc = initial_soc
        self.levels = levels
        self.decom_cost = decom_cost
        self.annual_om_cost_rate = annual_om_cost_rate
        self.annual_degr_rate = annual_degr_rate
        self.cap = cap
        self.loan_years = loan_years
        self.loan_payment = 0
        self.rem_loan = 0
        self.om_cost = om_cost
        self.rem_life = rem_life
        self.cur_eff = cur_eff
        self.cur_dod = cur_dod

    def install_new(self, new_cap):
        self.cap = new_cap
        self.loan_payment = - payments_per_year * np.pmt(interest_rate / payments_per_year, self.loan_years * payments_per_year, self.price_list[self.price_state]) * new_cap
        self.rem_loan = self.loan_payment * self.loan_years
        self.om_cost = years_in_period * self.annual_om_cost_rate * self.price_list[self.price_state] * new_cap
        self.rem_life = self.life_list[self.life_state] if new_cap != 0 else 0
        self.cur_eff = self.eff_list[self.eff_state] if new_cap != 0 else 0
        self.cur_dod = self.dod_list[self.dod_state] if new_cap != 0 else 0

    def decrease_life(self):
        if self.rem_life <= years_in_period:
            self.cap = 0
            self.loan_payment = 0
            self.rem_loan = 0
        self.rem_life = max(self.rem_life-years_in_period, 0)
        self.cap = self.cap * (1 - years_in_period * self.annual_degr_rate)

    def state_transition(self, feature):
        if random.uniform(0, 1) < getattr(self, feature + '_probs')[getattr(self, feature + '_state')]:
            setattr(self, feature + '_state', getattr(self, feature + '_state') + 1)

""" microgrid facilities """
fcnt, fcrit_load, fvoll, fev_profile, fnum_ev, flolp_constraint, fccp_constraint, fkkt_multiplier = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

""" hospital data """
key = 'hospital'
fcnt[key] = 2
fcrit_load[key] = 0.8
fvoll[key] = 25
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [0] * 24
fnum_ev[key] = 0
flolp_constraint[key] = 0.2
fccp_constraint[key] = 0.8
fkkt_multiplier[key] = 10**6

""" outpatient data """
key = 'outpatient'
fcnt[key] = 2
fcrit_load[key] = 0.8
fvoll[key] = 19
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [0] * 24
fnum_ev[key] = 0
flolp_constraint[key] = 0.2
fccp_constraint[key] = 0.8
fkkt_multiplier[key] = 3*10**5

""" supermarket data """
key = 'supermarket'
fcnt[key] = 3
fcrit_load[key] = 0.6
fvoll[key] = 10
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [0] * 24
fnum_ev[key] = 0
flolp_constraint[key] = 0.5
fccp_constraint[key] = 0.5
fkkt_multiplier[key] = 10**5

""" hotel data """
key = 'hotel'
fcnt[key] = 3
fcrit_load[key] = 0.5
fvoll[key] = 9
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [x / 2 for x in [9, 9, 9, 9, 9, 9, 6.67, 4.33, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4.33, 6.67, 9, 9]]  # data coming from Prof. Jafari's research work on charging profiles for various facilities (Building 2)
fnum_ev[key] = 20
flolp_constraint[key] = 0.5
fccp_constraint[key] = 0.5
fkkt_multiplier[key] = 10**5

""" office data """
key = 'office'
fcnt[key] = 5
fcrit_load[key] = 0.5
fvoll[key] = 8
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [x / 2 for x in [2, 2, 2, 2, 2, 2, 2, 2, 4.33, 6.67, 9, 9, 9, 9, 9, 9, 9, 6.67, 4.33, 2, 2, 2, 2, 2]]  # data coming from Prof. Jafari's research work on charging profiles for various facilities (Office)
fnum_ev[key] = 10
flolp_constraint[key] = 0.5
fccp_constraint[key] = 0.5
fkkt_multiplier[key] = 10**5

""" school data """
key = 'school'
fcnt[key] = 3
fcrit_load[key] = 0.4
fvoll[key] = 7
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [0] * 24
fnum_ev[key] = 0
flolp_constraint[key] = 0.5
fccp_constraint[key] = 0.5
fkkt_multiplier[key] = 10**5

""" restaturant data """
key = 'restaurant'
fcnt[key] = 7
fcrit_load[key] = 0.9
fvoll[key] = 6
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [0] * 24
fnum_ev[key] = 0
flolp_constraint[key] = 0.5
fccp_constraint[key] = 0.5
fkkt_multiplier[key] = 10**5

""" residential data """
key = 'residential'
fcnt[key] = 300
fcrit_load[key] = 0.3
fvoll[key] = 5
assert fvoll[key] == min(fvoll.values()), "Facilities should be initialized in a voll-sorted fashion!"
fev_profile[key] = [x / 2 for x in [8, 8, 8, 8, 8, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 6]]  # data coming from Prof. Jafari's research work on charging profiles for various facilities (Apartment)
fnum_ev[key] = 2
flolp_constraint[key] = 0.5
fccp_constraint[key] = 0.5
fkkt_multiplier[key] = 10**5

""" power plants """
pprice_list, pprice_probs, pprice_state, plife_list, plife_probs, plife_state, peff_list, peff_probs, peff_state, plevels, pdecom_cost, pannual_om_cost_rate, pcap, ploan_years, pom_cost, prem_life, pcur_eff = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

""" solar data """
key = 'solar'
pprice_list[key] = [357.64675326, 349.64214247, 341.63753168, 333.63292089, 325.6283101, 317.62369932, 309.61908853, 301.61447774, 293.60986695, 285.60525616, 277.60064538, 269.59603459, 261.5914238, 253.58681301, 245.58220222, 237.57759144, 229.57298065, 221.56836986, 213.56375907, 205.55914829]     # remember that these are costs per solar panel, NOT cost per kw
pprice_list[key] = pprice_list[key][0::years_in_period]
pprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
pprice_state[key] = 0
plife_list[key] = [33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42]
plife_list[key] = plife_list[key][0::years_in_period]
plife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
plife_state[key] = 0
peff_list[key] = [0.17068049999999999, 0.18056824999999999, 0.19045600000000001, 0.20034375000000001, 0.21023150000000002, 0.22011925000000002, 0.23000700000000002, 0.23989475000000002, 0.24978250000000002, 0.25967025000000005, 0.269558, 0.27944575000000005, 0.2893335, 0.29922125000000005, 0.3091090000000001, 0.31899675000000005, 0.3288845000000001, 0.3387722500000001, 0.3486600000000001, 0.35854775000000005]
peff_list[key] = peff_list[key][0::years_in_period]
peff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
peff_state[key] = 0
plevels[key] = [2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000]
pdecom_cost[key] = 17.225
pannual_om_cost_rate[key] = 0.0063
pcap[key] = 0
ploan_years[key] = loan_horizon
pom_cost[key] = years_in_period * pannual_om_cost_rate[key] * pprice_list[key][pprice_state[key]] * pcap[key]
prem_life[key] = plife_list[key][plife_state[key]] if pcap[key] > 0 else 0
pcur_eff[key] = peff_list[key][peff_state[key]] if pcap[key] > 0 else 0

""" onshore wind data """
key = 'onshore wind'
pprice_list[key] = [167023.22820896, 163479.42728742, 159935.62636588, 156391.82544434, 152848.02452279, 149304.22360125, 145760.42267971, 142216.62175816, 138672.82083662, 135129.01991508, 131585.21899353, 128041.41807199, 124497.61715045, 120953.81622891, 117410.01530736, 113866.21438582, 110322.41346428, 106778.61254273, 103234.81162119 , 99691.01069965]     # remember that these are costs per wind turbine, NOT cost per kw
pprice_list[key] = pprice_list[key][0::years_in_period]
pprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
pprice_state[key] = 0
plife_list[key] = [17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26]
plife_list[key] = plife_list[key][0::years_in_period]
plife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
plife_state[key] = 0
peff_list[key] = [0.48, 0.48, 0.48, 0.48, 0.49, 0.49, 0.49, 0.49, 0.49, 0.50, 0.50, 0.50, 0.50, 0.51, 0.51, 0.51, 0.51, 0.52, 0.52, 0.52, 0.52]
peff_list[key] = peff_list[key][0::years_in_period]
assert all([x <= 0.593 for x in peff_list[key]]), "Betz's law"
peff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
peff_state[key] = 0
plevels[key] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pdecom_cost[key] = 5737.940464592778
pannual_om_cost_rate[key] = 0.007
pcap[key] = 0
ploan_years[key] = loan_horizon
pom_cost[key] = years_in_period * pannual_om_cost_rate[key] * pprice_list[key][pprice_state[key]] * pcap[key]
prem_life[key] = plife_list[key][plife_state[key]] if pcap[key] > 0 else 0
pcur_eff[key] = peff_list[key][peff_state[key]] if pcap[key] > 0 else 0

""" offshore wind data """
key = 'offshore wind'
pprice_list[key] = [736444.55558858, 729724.32356109, 723004.09153359, 716283.8595061, 709563.6274786, 702843.39545111, 696123.16342361, 689402.93139612, 682682.69936862, 675962.46734113, 669242.23531364, 662522.00328614, 655801.77125865, 649081.53923115, 642361.30720366, 635641.07517616, 628920.84314867, 622200.61112117, 615480.37909368, 608760.14706618]     # remember that these are costs per wind turbine, NOT cost per kw
pprice_list[key] = pprice_list[key][0::years_in_period]
pprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
pprice_state[key] = 0
plife_list[key] = [17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26]
plife_list[key] = plife_list[key][0::years_in_period]
plife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
plife_state[key] = 0
peff_list[key] = [0.48, 0.48, 0.48, 0.48, 0.49, 0.49, 0.49, 0.49, 0.49, 0.50, 0.50, 0.50, 0.50, 0.51, 0.51, 0.51, 0.51, 0.52, 0.52, 0.52, 0.52]
peff_list[key] = peff_list[key][0::years_in_period]
assert all([x <= 0.593 for x in peff_list[key]]), "Betz's law"
peff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
peff_state[key] = 0
plevels[key] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pdecom_cost[key] = 36385.233540973386
pannual_om_cost_rate[key] = 0.014
pcap[key] = 0
ploan_years[key] = loan_horizon
pom_cost[key] = years_in_period * pannual_om_cost_rate[key] * pprice_list[key][pprice_state[key]] * pcap[key]
prem_life[key] = plife_list[key][plife_state[key]] if pcap[key] > 0 else 0
pcur_eff[key] = peff_list[key][peff_state[key]] if pcap[key] > 0 else 0

""" diesel data """
key = 'diesel'
pprice_list[key] = [800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800]
pprice_list[key] = pprice_list[key][0::years_in_period]
pprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
pprice_state[key] = 0
plife_list[key] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
plife_list[key] = plife_list[key][0::years_in_period]
plife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
plife_state[key] = 0
peff_list[key] = [0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.44, 0.44, 0.44, 0.44, 0.45, 0.45, 0.45, 0.45, 0.45, 0.46, 0.46, 0.46, 0.46, 0.46]
peff_list[key] = peff_list[key][0::years_in_period]
peff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
peff_state[key] = 0
plevels[key] = [100, 400, 700, 1_000, 1_300, 1_600, 1_900, 2_200, 2_500, 2_800]
pdecom_cost[key] = 31
pannual_om_cost_rate[key] = 0.04375
pcap[key] = 0
ploan_years[key] = loan_horizon
pom_cost[key] = years_in_period * pannual_om_cost_rate[key] * pprice_list[key][pprice_state[key]] * pcap[key]
prem_life[key] = plife_list[key][plife_state[key]] if pcap[key] > 0 else 0
pcur_eff[key] = peff_list[key][peff_state[key]] if pcap[key] > 0 else 0

""" hydro data """
key = 'hydro'
pprice_list[key] = [1518.19907009, 1544.39814019, 1570.59721028, 1596.79628038, 1622.99535047, 1649.19442056, 1675.39349066, 1701.59256075, 1727.79163085, 1753.99070094, 1780.18977104, 1806.38884113, 1832.58791122, 1858.78698132, 1884.98605141, 1911.18512151, 1937.3841916, 1963.58326169, 1989.78233179, 2015.98140188]
pprice_list[key] = pprice_list[key][0::years_in_period]
pprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
pprice_state[key] = 0
plife_list[key] = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
plife_list[key] = plife_list[key][0::years_in_period]
plife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
plife_state[key] = 0
peff_list[key] = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
peff_list[key] = peff_list[key][0::years_in_period]
peff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
peff_state[key] = 0
plevels[key] = [100, 400, 700, 1_000, 1_300, 1_600, 1_900, 2_200, 2_500, 2_800]
pdecom_cost[key] = 303.63981401800004
pannual_om_cost_rate[key] = 0.025
pcap[key] = 0
ploan_years[key] = loan_horizon
pom_cost[key] = years_in_period * pannual_om_cost_rate[key] * pprice_list[key][pprice_state[key]] * pcap[key]
prem_life[key] = plife_list[key][plife_state[key]] if pcap[key] > 0 else 0
pcur_eff[key] = peff_list[key][peff_state[key]] if pcap[key] > 0 else 0

""" storage units """
sprice_list, sprice_probs, sprice_state, slife_list, slife_probs, slife_state, seff_list, seff_probs, seff_state, sdod_list, sdod_probs, sdod_state, sinitial_soc, slevels, sdecom_cost, sannual_om_cost_rate, sannual_degr_rate, scap, sloan_years, som_cost, srem_life, scur_eff, scur_dod = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

""" lithium-ion data """
key = 'lithium-ion'
sprice_list[key] = [470.0, 449.4736842105263, 428.9473684210526, 408.42105263157896, 387.89473684210526, 367.36842105263156, 346.8421052631579, 326.3157894736842, 305.7894736842105, 285.2631578947368, 264.7368421052631, 244.21052631578948, 223.68421052631578, 203.15789473684208, 182.63157894736844, 162.10526315789474, 141.57894736842104, 121.05263157894734, 100.52631578947364, 80.0]
if key in settings['increasing_storage_prices']:
    sprice_list[key] = [470.0, 478.94736842105266, 487.89473684210526, 496.8421052631579, 505.7894736842105, 514.7368421052631, 523.6842105263158, 532.6315789473684, 541.578947368421, 550.5263157894736, 559.4736842105264, 568.421052631579, 577.3684210526316, 586.3157894736842, 595.2631578947369, 604.2105263157895, 613.1578947368421, 622.1052631578948, 631.0526315789474, 640.0]
sprice_list[key] = sprice_list[key][0::years_in_period]
sprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sprice_state[key] = 0
slife_list[key] = [12, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23]
slife_list[key] = slife_list[key][0::years_in_period]
slife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
slife_state[key] = 0
seff_list[key] = [0.94, 0.94, 0.94, 0.94, 0.95, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98]
seff_list[key] = seff_list[key][0::years_in_period]
seff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
seff_state[key] = 0
sdod_list[key] = [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
sdod_list[key] = sdod_list[key][0::years_in_period]
sdod_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sdod_state[key] = 0
sinitial_soc[key] = 1
slevels[key] = [1_500, 3_500, 5_500, 7_500, 9_500, 11_500, 13_500, 15_500, 17_500, 19_500]
sdecom_cost[key] = 220
sannual_om_cost_rate[key] = 0.025
sannual_degr_rate[key] = 0.0171
scap[key] = 0
sloan_years[key] = loan_horizon
som_cost[key] = years_in_period * sannual_om_cost_rate[key] * sprice_list[key][sprice_state[key]] * scap[key]
srem_life[key] = slife_list[key][slife_state[key]] if scap[key] > 0 else 0
scur_eff[key] = seff_list[key][seff_state[key]] if scap[key] > 0 else 0
scur_dod[key] = sdod_list[key][sdod_state[key]] if scap[key] > 0 else 0

""" lead acid data """
key = 'lead acid'
sprice_list[key] = [260.0, 248.94736842105263, 237.89473684210526, 226.8421052631579, 215.78947368421052, 204.73684210526315, 193.68421052631578, 182.63157894736844, 171.57894736842104, 160.5263157894737, 149.4736842105263, 138.42105263157896, 127.36842105263159, 116.31578947368422, 105.26315789473685, 94.21052631578948, 83.15789473684211, 72.10526315789474, 61.05263157894737, 50.0]
if key in settings['increasing_storage_prices']:
    sprice_list[key] = [260.0, 266.8421052631579, 273.6842105263158, 280.5263157894737, 287.36842105263156, 294.2105263157895, 301.0526315789474, 307.89473684210526, 314.7368421052632, 321.57894736842104, 328.42105263157896, 335.2631578947368, 342.10526315789474, 348.9473684210526, 355.7894736842105, 362.63157894736844, 369.4736842105263, 376.3157894736842, 383.1578947368421, 390.0]
sprice_list[key] = sprice_list[key][0::years_in_period]
sprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sprice_state[key] = 0
slife_list[key] = [9, 10, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20]
slife_list[key] = slife_list[key][0::years_in_period]
slife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
slife_state[key] = 0
seff_list[key] = [0.80, 0.80, 0.80, 0.80, 0.81, 0.81, 0.81, 0.81, 0.82, 0.82, 0.82, 0.82, 0.83, 0.83, 0.83, 0.83, 0.84, 0.84, 0.84, 0.84]
seff_list[key] = seff_list[key][0::years_in_period]
seff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
seff_state[key] = 0
sdod_list[key] = [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]
sdod_list[key] = sdod_list[key][0::years_in_period]
sdod_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sdod_state[key] = 0
sinitial_soc[key] = 1
slevels[key] = [1_500, 3_500, 5_500, 7_500, 9_500, 11_500, 13_500, 15_500, 17_500, 19_500]
sdecom_cost[key] = 88
sannual_om_cost_rate[key] = 0.025
sannual_degr_rate[key] = 0.0171
scap[key] = 0
sloan_years[key] = loan_horizon
som_cost[key] = years_in_period * sannual_om_cost_rate[key] * sprice_list[key][sprice_state[key]] * scap[key]
srem_life[key] = slife_list[key][slife_state[key]] if scap[key] > 0 else 0
scur_eff[key] = seff_list[key][seff_state[key]] if scap[key] > 0 else 0
scur_dod[key] = sdod_list[key][sdod_state[key]] if scap[key] > 0 else 0

""" vanadium redox """
key = 'vanadium redox'
sprice_list[key] = [400.0, 383.1578947368421, 366.3157894736842, 349.4736842105263, 332.63157894736844, 315.7894736842105, 298.9473684210526, 282.10526315789474, 265.2631578947369, 248.42105263157896, 231.57894736842107, 214.73684210526318, 197.89473684210526, 181.05263157894737, 164.21052631578948, 147.3684210526316, 130.5263157894737, 113.68421052631578, 96.84210526315792, 80.0]
if key in settings['increasing_storage_prices']:
    sprice_list[key] = [400.0, 407.89473684210526, 415.7894736842105, 423.6842105263158, 431.57894736842104, 439.4736842105263, 447.36842105263156, 455.2631578947368, 463.1578947368421, 471.0526315789474, 478.9473684210526, 486.8421052631579, 494.7368421052631, 502.63157894736844, 510.5263157894737, 518.421052631579, 526.3157894736842, 534.2105263157895, 542.1052631578948, 550.0]
sprice_list[key] = sprice_list[key][0::years_in_period]
sprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sprice_state[key] = 0
slife_list[key] = [13, 14, 15, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24]
slife_list[key] = slife_list[key][0::years_in_period]
slife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
slife_state[key] = 0
seff_list[key] = [0.70, 0.71, 0.72, 0.73, 0.74, 0.74, 0.75, 0.75, 0.76, 0.76, 0.77, 0.77, 0.78, 0.78, 0.79, 0.79, 0.80, 0.80, 0.81, 0.81]
seff_list[key] = seff_list[key][0::years_in_period]
seff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
seff_state[key] = 0
sdod_list[key] = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
sdod_list[key] = sdod_list[key][0::years_in_period]
sdod_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sdod_state[key] = 0
sinitial_soc[key] = 1
slevels[key] = [1_500, 3_500, 5_500, 7_500, 9_500, 11_500, 13_500, 15_500, 17_500, 19_500]
sdecom_cost[key] = 300
sannual_om_cost_rate[key] = 0.025
sannual_degr_rate[key] = 0.0171
scap[key] = 0
sloan_years[key] = loan_horizon
som_cost[key] = years_in_period * sannual_om_cost_rate[key] * sprice_list[key][sprice_state[key]] * scap[key]
srem_life[key] = slife_list[key][slife_state[key]] if scap[key] > 0 else 0
scur_eff[key] = seff_list[key][seff_state[key]] if scap[key] > 0 else 0
scur_dod[key] = sdod_list[key][sdod_state[key]] if scap[key] > 0 else 0

""" flywheel """
key = 'flywheel'
sprice_list[key] = [3100.0, 2989.4736842105262, 2878.9473684210525, 2768.421052631579, 2657.8947368421054, 2547.3684210526317, 2436.842105263158, 2326.315789473684, 2215.7894736842104, 2105.2631578947367, 1994.7368421052631, 1884.2105263157894, 1773.6842105263158, 1663.157894736842, 1552.6315789473683, 1442.1052631578948, 1331.578947368421, 1221.0526315789473, 1110.5263157894738, 1000.0]
if key in settings['increasing_storage_prices']:
    sprice_list[key] = [3100.0, 3121.0526315789475, 3142.1052631578946, 3163.157894736842, 3184.2105263157896, 3205.2631578947367, 3226.315789473684, 3247.3684210526317, 3268.421052631579, 3289.4736842105262, 3310.5263157894738, 3331.5789473684213, 3352.6315789473683, 3373.684210526316, 3394.7368421052633, 3415.7894736842104, 3436.842105263158, 3457.8947368421054, 3478.9473684210525, 3500.0]
sprice_list[key] = sprice_list[key][0::years_in_period]
sprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sprice_state[key] = 0
slife_list[key] = [20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30]
slife_list[key] = slife_list[key][0::years_in_period]
slife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
slife_state[key] = 0
seff_list[key] = [0.83, 0.83, 0.84, 0.84, 0.85, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.89, 0.90, 0.90, 0.91, 0.91, 0.92, 0.92]
seff_list[key] = seff_list[key][0::years_in_period]
seff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
seff_state[key] = 0
sdod_list[key] = [0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86]
sdod_list[key] = sdod_list[key][0::years_in_period]
sdod_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sdod_state[key] = 0
sinitial_soc[key] = 1
slevels[key] = [1_500, 3_500, 5_500, 7_500, 9_500, 11_500, 13_500, 15_500, 17_500, 19_500]
sdecom_cost[key] = 50
sannual_om_cost_rate[key] = 0.025
sannual_degr_rate[key] = 0
scap[key] = 0
sloan_years[key] = loan_horizon
som_cost[key] = years_in_period * sannual_om_cost_rate[key] * sprice_list[key][sprice_state[key]] * scap[key]
srem_life[key] = slife_list[key][slife_state[key]] if scap[key] > 0 else 0
scur_eff[key] = seff_list[key][seff_state[key]] if scap[key] > 0 else 0
scur_dod[key] = sdod_list[key][sdod_state[key]] if scap[key] > 0 else 0

""" pumped hydro """
key = 'pumped hydro'
sprice_list[key] = [1000.0, 989.4736842105264, 978.9473684210526, 968.421052631579, 957.8947368421052, 947.3684210526316, 936.8421052631579, 926.3157894736842, 915.7894736842105, 905.2631578947369, 894.7368421052631, 884.2105263157895, 873.6842105263158, 863.1578947368421, 852.6315789473684, 842.1052631578948, 831.578947368421, 821.0526315789473, 810.5263157894736, 800.0]
if key in settings['increasing_storage_prices']:
    sprice_list[key] = [1000.0, 1031.578947368421, 1063.157894736842, 1094.7368421052631, 1126.3157894736842, 1157.8947368421052, 1189.4736842105262, 1221.0526315789473, 1252.6315789473683, 1284.2105263157896, 1315.7894736842104, 1347.3684210526317, 1378.9473684210525, 1410.5263157894738, 1442.1052631578948, 1473.6842105263158, 1505.2631578947369, 1536.842105263158, 1568.421052631579, 1600.0]
sprice_list[key] = sprice_list[key][0::years_in_period]
sprice_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sprice_state[key] = 0
slife_list[key] = [60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63]
slife_list[key] = slife_list[key][0::years_in_period]
slife_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
slife_state[key] = 0
seff_list[key] = [0.80, 0.80, 0.80, 0.80, 0.80, 0.81, 0.81, 0.81, 0.81, 0.81, 0.82, 0.82, 0.82, 0.82, 0.82, 0.83, 0.83, 0.83, 0.83, 0.83]
seff_list[key] = seff_list[key][0::years_in_period]
seff_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
seff_state[key] = 0
sdod_list[key] = [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
sdod_list[key] = sdod_list[key][0::years_in_period]
sdod_probs[key] = (decision_periods - 1) * [mc_transition_prob] + [0]
sdod_state[key] = 0
sinitial_soc[key] = 1
slevels[key] = [1_500, 3_500, 5_500, 7_500, 9_500, 11_500, 13_500, 15_500, 17_500, 19_500]
sdecom_cost[key] = 200
sannual_om_cost_rate[key] = 0.025
sannual_degr_rate[key] = 0
scap[key] = 0
sloan_years[key] = loan_horizon
som_cost[key] = years_in_period * sannual_om_cost_rate[key] * sprice_list[key][sprice_state[key]] * scap[key]
srem_life[key] = slife_list[key][slife_state[key]] if scap[key] > 0 else 0
scur_eff[key] = seff_list[key][seff_state[key]] if scap[key] > 0 else 0
scur_dod[key] = sdod_list[key][sdod_state[key]] if scap[key] > 0 else 0

""" initialize the facilities, power_plants and storage_units OrderedDict """
facilities, power_plants, storage_units = OrderedDict(), OrderedDict(), OrderedDict()
for facility in facilities_list:
    facilities[facility] = MGridFacility(fcnt[facility], fcrit_load[facility], fvoll[facility], fev_profile[facility], fnum_ev[facility], flolp_constraint[facility], fccp_constraint[facility], fkkt_multiplier[facility])
for power_plant in power_plants_list:
    power_plants[power_plant] = MGridPowerPlant(pprice_list[power_plant], pprice_probs[power_plant], pprice_state[power_plant], plife_list[power_plant], plife_probs[power_plant], plife_state[power_plant], peff_list[power_plant], peff_probs[power_plant], peff_state[power_plant], plevels[power_plant], pdecom_cost[power_plant], pannual_om_cost_rate[power_plant], pcap[power_plant], ploan_years[power_plant], pom_cost[power_plant], prem_life[power_plant], pcur_eff[power_plant])
for storage_unit in storage_units_list:
    storage_units[storage_unit] = MGridStorageUnit(sprice_list[storage_unit], sprice_probs[storage_unit], sprice_state[storage_unit], slife_list[storage_unit], slife_probs[storage_unit], slife_state[storage_unit], seff_list[storage_unit], seff_probs[storage_unit], seff_state[storage_unit], sdod_list[storage_unit], sdod_probs[storage_unit], sdod_state[storage_unit], sinitial_soc[storage_unit], slevels[storage_unit], sdecom_cost[storage_unit], sannual_om_cost_rate[storage_unit], sannual_degr_rate[storage_unit], scap[storage_unit], sloan_years[storage_unit], som_cost[storage_unit], srem_life[storage_unit], scur_eff[storage_unit], scur_dod[storage_unit])
num_facilities, num_power_plants, num_storage_units = len(facilities), len(power_plants), len(storage_units)
assert num_facilities == len(facilities_list), 'There are some facilities not entered in the dictionary!'
assert num_power_plants == len(power_plants_list), 'There are some power plants not entered in the dictionary!'
assert num_storage_units == len(storage_units_list), 'There are some storage units not entered in the dictionary!'

""" states definition """
start_state, states_map, states_scaler, cnt = [0], dict(), [decision_periods-1], 1
states_map['period'] = 0
external_features_power_plants, internal_features_power_plants = ['price', 'life', 'eff'], ['cap', 'rem_life', 'cur_eff']
external_features_storage_units, internal_features_storage_units = ['price', 'life', 'eff', 'dod'], ['cap', 'rem_life', 'cur_eff', 'cur_dod']
for power_plant in power_plants:
    for external_feature in external_features_power_plants:
        start_state.append(getattr(power_plants[power_plant], external_feature + '_list')[getattr(power_plants[power_plant], external_feature + '_state')])
        states_map[(power_plant, external_feature)] = cnt
        states_scaler.append(max(getattr(power_plants[power_plant], external_feature + '_list')))
        cnt += 1
    for internal_feature in internal_features_power_plants:
        start_state.append(getattr(power_plants[power_plant], internal_feature))
        states_map[(power_plant, internal_feature)] = cnt
        if internal_feature == 'cap':
            states_scaler.append(max(power_plants[power_plant].levels))
        elif internal_feature == 'rem_life':
            states_scaler.append(max(power_plants[power_plant].life_list))
        elif internal_feature == 'cur_eff':
            states_scaler.append(1)
        else:
            raise ValueError('No scaler for this feature!')
        cnt += 1
for storage_unit in storage_units:
    for external_feature in external_features_storage_units:
        start_state.append(getattr(storage_units[storage_unit], external_feature + '_list')[getattr(storage_units[storage_unit], external_feature + '_state')])
        states_map[(storage_unit, external_feature)] = cnt
        states_scaler.append(max(getattr(storage_units[storage_unit], external_feature + '_list')))
        cnt += 1
    for internal_feature in internal_features_storage_units:
        start_state.append(getattr(storage_units[storage_unit], internal_feature))
        states_map[(storage_unit, internal_feature)] = cnt
        if internal_feature == 'cap':
            states_scaler.append(max(storage_units[storage_unit].levels))
        elif internal_feature == 'rem_life':
            states_scaler.append(max(storage_units[storage_unit].life_list))
        elif internal_feature == 'cur_eff':
            states_scaler.append(1)
        elif internal_feature == 'cur_dod':
            states_scaler.append(1)
        else:
            raise ValueError('No scaler for this feature!')
        cnt += 1
start_state = np.array(start_state)
start_state = start_state.reshape(1,len(start_state))
states_dim = len(start_state[0])
def norm_state(s):
    return s / np.array(states_scaler)
# total_number_states = 
# example: state = [0, li_price, la_price, vr_price, fw_price, li_cap, la_cap, vr_cap, fw_cap]

""" actions definiton """
actions_map = [0]
for power_plant in power_plants:
    actions_map.append((power_plant, 0))
    for level in power_plants[power_plant].levels:
        actions_map.append((power_plant, level))
for storage_unit in storage_units:
    actions_map.append((storage_unit, 0))
    for level in storage_units[storage_unit].levels:
        actions_map.append((storage_unit, level))
actions_dim = len(actions_map)
actions = np.linspace(0, actions_dim-1, actions_dim)
actions = list(map(int, actions))
# total_number_states_actions = total_number_states * actions_dim
# actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# action_tuples = [0, (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
# example: action = 5, action_tuple = (1,2)

""" reset the environment to the initial conditions """
def reset_environment():
    for power_plant in power_plants:
        power_plants[power_plant].price_state = 0
        power_plants[power_plant].life_state = 0
        power_plants[power_plant].eff_state = 0
        power_plants[power_plant].om_cost = pom_cost[power_plant]
        power_plants[power_plant].cap = pcap[power_plant]
        power_plants[power_plant].loan_payment = 0
        power_plants[power_plant].rem_loan = 0
        power_plants[power_plant].rem_life = prem_life[power_plant]
        power_plants[power_plant].cur_eff = pcur_eff[power_plant]
        if settings['random_transition_probabilities']:
            power_plants[power_plant].price_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]
            power_plants[power_plant].life_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]
            power_plants[power_plant].eff_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]
    for storage_unit in storage_units:
        storage_units[storage_unit].price_state = 0
        storage_units[storage_unit].life_state = 0
        storage_units[storage_unit].eff_state = 0
        storage_units[storage_unit].dod_state = 0
        storage_units[storage_unit].om_cost = som_cost[storage_unit]
        storage_units[storage_unit].cap = scap[storage_unit]
        storage_units[storage_unit].loan_payment = 0
        storage_units[storage_unit].rem_loan = 0
        storage_units[storage_unit].rem_life = srem_life[storage_unit]
        storage_units[storage_unit].cur_eff = scur_eff[storage_unit]
        storage_units[storage_unit].cur_dod = scur_dod[storage_unit]
        if settings['random_transition_probabilities']:
            storage_units[storage_unit].price_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]
            storage_units[storage_unit].life_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]
            storage_units[storage_unit].eff_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]
            storage_units[storage_unit].dod_probs = (decision_periods - 1) * [random.uniform(0,1)] + [0]