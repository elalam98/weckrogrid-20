"""
Created on Mon Mar 16 16:20:13 2020
Outage simulation
@author: Stamatis
"""

""" prepare input for neural network """
def prepare_nn_input(p, start):
    res = [dict() for _ in range(len(facilities))]
    for i, facility in enumerate(facilities):
        if i == 0:
            for wp in settings['weather_predictors']:
                res[i][wp] = meteo[wp][start-settings['input_dim']:start]
        res[i]['Demand'] = demand[p][i][start-settings['input_dim']:start]
    return res

""" value of lost load as a function of outage duration """
def compute_voll(a, b, x):
    return a + math.exp(b*x) if settings['use_exponential_voll'] else a
            
""" helper function """
def outage_simulation_helper(period, runs):
    
    """ outage generation """
    outage_time, outage_duration = list(), list()
    for i in range(runs):
        temp_time, temp_duration, time_sum = list(), list(), 0
        time_sum = round(np.random.exponential(scale_total))
        while time_sum < settings['input_dim']:
            time_sum = round(np.random.exponential(scale_total))
        #if time_sum > years_in_period * HRS_IN_YEAR: 
            #print('\nThe run number ' + str(i+1) + ' has 0 outages!\n')
        while time_sum <= years_in_period * HRS_IN_YEAR:
            temp_time.append(time_sum)
            if np.random.uniform(0, 1) < prob_extreme: 
                temp_duration.append(1 + np.random.poisson(caidi_extreme - 1))
            else:
                temp_duration.append(1 + np.random.poisson(caidi_normal - 1))
            time_sum += round(np.random.exponential(scale_total))
        outage_time.append(temp_time)
        outage_duration.append(temp_duration)
       
    """ tech initialization """
    tech_max, tech_dod_temp, tech_eff_temp = [], [], []
    for storage_unit in storage_units:
        tech_max.append(storage_units[storage_unit].cap)
        tech_dod_temp.append(storage_units[storage_unit].cur_dod)
        tech_eff_temp.append(storage_units[storage_unit].cur_eff)
    tech_min = [a * (1 - b) for a, b in zip(tech_max, tech_dod_temp)]
    tech_sum_discharge, tech_sum_charge = [a * b * c for a, b, c in zip(tech_max, tech_eff_temp, tech_dod_temp)], [d * (1 / e) * f  if e != 0 else 0 for d, e, f in zip(tech_max, tech_eff_temp, tech_dod_temp)] # compute the eff (or 1 / eff) * max * dod values for every technology
    sum_tech_sum_discharge, sum_tech_sum_charge = sum(tech_sum_discharge), sum(tech_sum_charge)
    if sum_tech_sum_discharge != 0 and sum_tech_sum_charge != 0:
        tech_rate_discharge, tech_rate_charge = [x / sum_tech_sum_discharge for x in tech_sum_discharge],  [y / sum_tech_sum_charge for y in tech_sum_charge] # compute the rate at which every technology should contribute to load discharge or charge
    array_opr_cost, array_out_cost, array_lolp, array_ccp = runs * [0], runs * [0], runs * [num_facilities * [0]], runs * [num_facilities * [0]] # the final arrays containing results for every simulation run
        
    """ loop over every simulation run """
    for run in range(runs):
        run_opr_cost, run_out_cost, run_lolp, run_ccp = 0, 0, num_facilities * [0], num_facilities * [0] # results for each simulation run
        free_total_production = 0
        for power_plant in power_plants:  # calculate operating costs for when there are no outages
            free_total_production += power_plants[power_plant].cur_eff * power_plants[power_plant].cap * power_plants[power_plant].power
        run_opr_cost = - min(free_total_production, demand_per_period[period]) * (HRS_IN_YEAR * years_in_period - sum(outage_duration[run])) * ELECTRICITY_PRICE_PER_KWH
        how_many_outages = len(outage_time[run])
        if how_many_outages == 0:
            array_lolp[run], array_ccp[run], array_opr_cost[run], array_out_cost[run] = num_facilities * [0], num_facilities * [1], run_opr_cost, 0
            continue
            
        """ loop over every outage in run """
        for timeout, duration in zip(outage_time[run], outage_duration[run]):
            hourly_lolp, hours_without_power = num_facilities * [0], num_facilities * [0]
            tech_energy = [storage_units[storage_unit].initial_soc * storage_units[storage_unit].cap for storage_unit in storage_units] # energy stored in each storage technology
            tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)] # available energy in each storage technology
                
            """ loop over every hour in outage """
            for hour in range(duration):
                cum_demand = []
                for i in range(num_facilities):
                    cum_demand.append(demand[period][i][timeout+hour])
                cum_free_production, cum_pay_production = 0, 0
                for power_plant in power_plants:
                    if power_plants[power_plant].power != 0:
                        cum_free_production += power_plants[power_plant].prod[timeout+hour] * power_plants[power_plant].cur_eff * power_plants[power_plant].cap
                    else:
                        cum_pay_production += power_plants[power_plant].prod[timeout+hour] * power_plants[power_plant].cur_eff * power_plants[power_plant].cap
                sum_tech_avail_energy, sum_cum_demand = sum(tech_avail_energy), sum(cum_demand)
                """ get nn demand forecast and reduce production and available energy if necessary """
                if settings['use_forecasted_demand']:
                    x_input = prepare_nn_input(period, timeout+hour)
                    cum_forecasted_demand = model_predict(nn_model, x_input)
                    cum_forecasted_demand = [x * (1 + settings['reserve_margin_rate']) for x in cum_forecasted_demand]
                    sum_cum_forecasted_demand = sum(cum_forecasted_demand)
                    if cum_free_production + cum_pay_production + sum_tech_avail_energy > sum_cum_forecasted_demand:
                        difference = cum_free_production + cum_pay_production + sum_tech_avail_energy - sum_cum_forecasted_demand
                        rate_free_production, rate_pay_production, rate_avail_energy = cum_free_production / (cum_free_production + cum_pay_production + sum_tech_avail_energy), cum_pay_production / (cum_free_production + cum_pay_production + sum_tech_avail_energy), sum_tech_avail_energy / (cum_free_production + cum_pay_production + sum_tech_avail_energy)
                        cum_free_production -= difference * rate_free_production
                        cum_pay_production -= difference * rate_pay_production
                        sum_tech_avail_energy -= difference * rate_avail_energy
                if cum_free_production >= sum(cum_demand): # renewables production enough to satisfy the demand
                    energy_amount = cum_free_production - sum_cum_demand
                    if sum_tech_sum_discharge != 0 and sum_tech_sum_charge != 0:
                        tech_energy = [min(a + b * energy_amount * c, d) for a, b, c, d in zip(tech_energy, tech_rate_charge, tech_eff_temp, tech_max)]
                    tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                elif cum_free_production + cum_pay_production >= sum_cum_demand: # renewables and diesel production enough to satisfy the demand
                    run_opr_cost += (sum_cum_demand - cum_free_production) * DIESEL_PRICE_PER_KWH
                elif cum_free_production + cum_pay_production + sum_tech_avail_energy >= sum_cum_demand: # renewables and diesel production together with storage devices enough to satisfy the demand
                    run_opr_cost += cum_pay_production * DIESEL_PRICE_PER_KWH
                    energy_amount = sum_cum_demand - cum_free_production - cum_pay_production
                    if sum_tech_sum_discharge != 0 and sum_tech_sum_charge != 0:
                        tech_energy = [a - (b * energy_amount) / c if c != 0 else 0 for a, b, c in zip(tech_energy, tech_rate_discharge, tech_eff_temp)]
                    tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                else: # microgrid not able to satisfy the demand
                    jcount, temp_free_prod, temp_pay_prod = -1, cum_free_production, cum_pay_production
                    while temp_free_prod + temp_pay_prod + sum_tech_avail_energy < sum(cum_demand[:jcount]):
                        jcount -= 1
                        if jcount == -num_facilities: break
                    facilities_satisfied = num_facilities + jcount
                    for i, facility in enumerate(facilities):
                        if i < facilities_satisfied:
                            hours_without_power[i] = 0
                            if temp_free_prod >= demand[period][i][timeout+hour]:
                                temp_free_prod -= demand[period][i][timeout+hour]
                            elif temp_free_prod + temp_pay_prod >= demand[period][i][timeout+hour]:
                                temp_free_prod = 0
                                temp_pay_prod -= demand[period][i][timeout+hour] - temp_free_prod
                                run_opr_cost += (demand[period][i][timeout+hour] - temp_free_prod) * DIESEL_PRICE_PER_KWH
                            else:
                                energy_amount = demand[period][i][timeout+hour] - temp_free_prod - temp_pay_prod
                                run_opr_cost += temp_pay_prod * DIESEL_PRICE_PER_KWH
                                temp_free_prod, temp_pay_prod = 0, 0
                                if sum_tech_sum_discharge != 0 and sum_tech_sum_charge != 0:
                                    tech_energy = [a - (b * energy_amount) / c if c != 0 else 0 for a, b, c in zip(tech_energy, tech_rate_discharge, tech_eff_temp)]
                                tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                        else:
                            hours_without_power[i] += 1
                            hourly_lolp[i] += 1
                            run_out_cost += demand[period][i][timeout+hour] * compute_voll(facilities[facility].voll, settings['exponential_multiplier'], hours_without_power[i])
                    energy_amount = temp_free_prod
                    if sum_tech_sum_discharge != 0 and sum_tech_sum_charge != 0:
                        tech_energy = [min(a + b * energy_amount * c, d) for a, b, c, d in zip(tech_energy, tech_rate_charge, tech_eff_temp, tech_max)]
                    tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                    """ end of loop over every hour in outage """
                        
            temp_lolp = [k / duration for k in hourly_lolp]
            temp_ccp = [1 if temp_lolp[i] <= facilities[facility].lolp_constraint else 0 for i, facility in enumerate(facilities)]
            run_lolp = [a + b for a, b in zip(run_lolp, temp_lolp)]
            run_ccp = [a + b for a, b in zip(run_ccp, temp_ccp)]
            """ end of loop over every outage in run """
                
        array_lolp[run] = [a / how_many_outages for a in run_lolp]
        array_ccp[run] = [b / how_many_outages for b in run_ccp]
        array_opr_cost[run] = run_opr_cost
        array_out_cost[run] = run_out_cost
        """ end of loop over every simulation run """
        
    lolp = np.mean(array_lolp, axis = 0) # final lolp statistic for each system configuration
    ccp = np.mean(array_ccp, axis = 0) # final ccp statistic for each system configuration
    opr_cost = np.mean(array_opr_cost)
    out_cost = np.mean(array_out_cost) + sum([years_in_period * facilities[facility].kkt_multiplier if ccp[i] < facilities[facility].ccp_constraint else 0 for i, facility in enumerate(facilities)]) # final cost statistic for each system configuration
    
    return opr_cost, out_cost, lolp, ccp