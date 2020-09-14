"""
Created on Fri Mar 8 17:04:19 2019
Import data
@author: Stamatis
"""

def read_electricity_data(start_date=settings['start_date'], end_date=settings['end_date']):
    total_electricity_data = []
    for facility in facilities:
        dataset = pd.read_csv(settings['filename_electricity'] + facility + '.csv')
        electricity_data = dataset[['Date/Time', 'Electricity:Facility [kW](Hourly)']]
        electricity_data = pd.DataFrame.drop_duplicates(electricity_data)
                
        electricity_data['Electricity:Facility [kW](Hourly)'] = electricity_data['Electricity:Facility [kW](Hourly)'].replace(' ', '') # remove empty spaces
        electricity_data['Electricity:Facility [kW](Hourly)'] = electricity_data['Electricity:Facility [kW](Hourly)'].replace('Null', '-1') # handle missing data
        try:
            data_median = median(list(map(float,electricity_data['Electricity:Facility [kW](Hourly)'])))
            electricity_data['Electricity:Facility [kW](Hourly)'] = electricity_data['Electricity:Facility [kW](Hourly)'].replace('-1', str(data_median))
        except:
            pass
        
        new_series = pd.Series(data = electricity_data['Electricity:Facility [kW](Hourly)'].values, index = pd.date_range(start='2002-01-01 01:00:00', end='2003-01-01 00:00:00', freq='1h'))
        data_reindexed = electricity_data.reindex(pd.date_range(start=start_date, end=end_date, freq='1h'))  
        data_reindexed['Electricity:Facility [kW](Hourly)'] = new_series
        new_df = data_reindexed.reset_index()
        new_df = new_df.drop(columns=['Date/Time'])
        new_df = new_df.rename(columns={'index': 'Date/Time'})
        
        new_df = new_df.rename(columns={'Electricity:Facility [kW](Hourly)': 'Demand', 'Date/Time': 'Date_Time'})
        new_df = new_df[(new_df.Date_Time >= start_date) & (new_df.Date_Time <= end_date)]
        new_df['Demand'] = facilities[facility].cnt * new_df['Demand']
            
        new_df = new_df.reset_index()
        new_df = new_df.drop(columns=['index'])
        new_df.loc[:, 'Facility'] = facility
        if settings['include_ev_load']:
            ev_additional_load_daily = [facilities[facility].num_ev * facilities[facility].cnt * x for x in facilities[facility].ev_profile]
            ev_additional_load_total = ev_additional_load_daily * DAYS_IN_YEAR
            new_df['Demand'] += ev_additional_load_total[:-1]
        total_electricity_data.append(new_df)
    
    return total_electricity_data

def read_weather_data(start_date=settings['start_date'], end_date=settings['end_date']):
    dataset = pd.read_csv(settings['filename_weather'] + '.csv')
    weather_data = dataset[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Temperature', 'GHI', 'Wind Speed']]
    weather_data.loc[:, 'Year'] = '2002'
    
    weather_data = pd.DataFrame.drop_duplicates(weather_data)

    weather_data['Date_Time'] = pd.to_datetime(weather_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    weather_data = weather_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
    weather_data = weather_data[(weather_data.Date_Time >= start_date) & (weather_data.Date_Time <= end_date)]
    
    """ handle missing values in weather dataframe """
    weather_data = weather_data.replace(-999,0)
    
    return weather_data

def interpolate_dataframe(df, frequency, min_datetime=settings['start_date'], max_datetime=settings['end_date']):
    df.index = df['Date_Time']
    df_reindexed = df.reindex(pd.date_range(start=min_datetime, end=max_datetime, freq=frequency))     
    df_interpolated = df_reindexed.interpolate(method='linear')
    new_df = df_interpolated.reset_index()
    new_df = new_df.drop(columns=['Date_Time'])
    new_df = new_df.rename(columns={'index': 'Date_Time'})
    
    return new_df.fillna(method='bfill')

def join_dataframes(df1, df2, based_on):  
    joint_df = pd.merge(df1, df2, how='inner', on=based_on)
    
    return joint_df.fillna(method='bfill')

""" read electricity and weather datasets """
electricity = read_electricity_data()
weather = read_weather_data()

""" interpolate and join dataframes """
w = interpolate_dataframe(weather, '30T')
e = []
for i, ele in enumerate(electricity):
    interpolated = interpolate_dataframe(electricity[i], '30T')
    if i == 0:
        interpolated = join_dataframes(interpolated, w, 'Date_Time')
    e.append(interpolated)
    
""" keep only 1-hour intervals """
delete_odd = [x for x in range(1,len(e[0]),2)]
single_df = []
for ele in e:
    ele = ele.drop(labels=delete_odd)
    single_df.append(ele)

""" create demand array """
for i, facility in enumerate(facilities):
    facilities[facility].yearly_data = single_df[i]['Demand'].values
for facility in facilities:
    facilities[facility].total_data = copy.deepcopy(facilities[facility].yearly_data)
for i in range(years_in_period):
    for facility in facilities:
        facilities[facility].total_data = np.append(facilities[facility].total_data, facilities[facility].yearly_data)
demand, demand_per_period = list(), list()
for i in range(decision_periods):
    to_append = list()
    for facility in facilities:
        to_append.append([(1 + settings['annual_demand_growth_rate_percent'] * years_in_period / 100) ** i * facilities[facility].crit_load * a for a in facilities[facility].total_data])
    demand_per_period.append(sum([np.mean(x) for x in to_append]))
    demand.append(to_append)

""" create weather array """
solar_ghi_orig, wind_speed_orig = single_df[0]['GHI'].values, single_df[0]['Wind Speed'].values
solar_ghi, wind_speed = [], []
meteo = dict()
for wp in settings['weather_predictors']:
    meteo[wp] = []
    for i in range(years_in_period+1):
        meteo[wp].extend(single_df[0][wp].values)
for i in range(years_in_period+1):
    solar_ghi.extend(solar_ghi_orig)
    wind_speed.extend(wind_speed_orig)
    
""" calculate solar production """
if 'solar' in power_plants_list:
    solar_cell_area, solar_cells_per_panel = 0.0232258, 72 # solar_area is the area of the solar cell
    # using solar_efficiency = 0.16, solar_cell_area = 0.0232258, solar_cells_per_panel = 72, then 20 solar panels output approximately 1kW
    power_plants['solar'].prod = [0.001 * solar_cell_area * solar_cells_per_panel * a for a in solar_ghi] # prod per solar panel without considering efficiency
    power_plants['solar'].power = np.mean(power_plants['solar'].prod)

""" calculate onshore wind production """
if 'onshore wind' in power_plants_list:
    wind_diameter, cut_in_speed, cut_out_speed = 44, 3, 22
    wind_area = math.pi * (wind_diameter / 2) ** 2 # wind_area is the rotor swept area
    # using wind_efficiency = 0.48, wind_diameter = 44, then 1 onshore wind turbine outputs approximately 37 kW
    power_plants['onshore wind'].prod = [0.001 * 0.5 * 1.25 * wind_area * a ** 3 if a >= cut_in_speed and a <= cut_out_speed else 0 for a in wind_speed]  # prod per onshore wind turbine without considering efficiency
    power_plants['onshore wind'].power = np.mean(power_plants['onshore wind'].prod)

""" calculate offshore wind production """
if 'offshore wind' in power_plants_list:
    wind_reduction = 0.2
    power_plants['offshore wind'].prod =  [0.001 * 0.5 * 1.25 * wind_area * (a * (1 / (1 - wind_reduction))) ** 3 if (a * (1 / (1 - wind_reduction))) >= cut_in_speed and (a * (1 / (1 - wind_reduction))) <= cut_out_speed else 0 for a in wind_speed]  # prod per offshore wind turbine without considering efficiency
    power_plants['offshore wind'].power = np.mean(power_plants['offshore wind'].prod)

""" calculate diesel production """
if 'diesel' in power_plants_list:
    power_plants['diesel'].prod = [1 for _ in solar_ghi]
    power_plants['diesel'].power = 0  # this basically means that diesel does not contribute to free production

""" calculate hydro production """
if 'hydro' in power_plants_list:
    power_plants['hydro'].prod = [1 for _ in solar_ghi]
    power_plants['hydro'].power = 0.5 # this is the capacity factor of hydropower

""" outage parameters """
if settings['superposed_scenario']: 
    saifi_extreme, saifi_normal, caidi_extreme, caidi_normal = 1 / 6, 0.84 / 6 + 0.89 / 6 + 0.76 / 6 + 1 / 6 + 1.34 / 6 + 1.1 / 6, 22.55, (1.65 + 1.42 + 1.95 + 1.46 + 1.7) / 5
else: 
    saifi_extreme, saifi_normal, caidi_extreme, caidi_normal = (1.84 + 0.89 + 0.76 + 1 + 1.34 + 1.1) / (6 * 2), (1.84 + 0.89 + 0.76 + 1 + 1.34 + 1.1) / (6 * 2), (22.55 + 1.65 + 1.42 + 1.95 + 1.46 + 1.7) / 6, (22.55 + 1.65 + 1.42 + 1.95 + 1.46 + 1.7) / 6
prob_extreme = saifi_extreme / (saifi_extreme + saifi_normal) # probability that an outage belongs to the extreme poisson process
prob_normal = 1 - prob_extreme
saifi_total = saifi_extreme + saifi_normal
scale_total_years = 1 / saifi_total
scale_total = scale_total_years * 8760 # scale parameter of the superposed poisson process converted in hours