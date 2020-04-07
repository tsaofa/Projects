# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:54:46 2020

@author: tsaof
"""

import pandas as pd
import numpy as np
import datetime as dt


def get_data(save = False, nrows = None):
    flights = pd.read_csv('flights.csv',nrows = nrows, dtype={'SCHEDULED_DEPARTURE': str,
                                               'DEPARTURE_TIME': str,
                                               'SCHEDULED_ARRIVAL': str,
                                               'ARRIVAL_TIME': str})
    
    columns_to_drop = ['YEAR', 'DAY','TAXI_OUT','WHEELS_OFF', 'SCHEDULED_TIME',
                       'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 
                       'TAXI_IN','CANCELLATION_REASON','AIR_SYSTEM_DELAY', 
                       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY']
    
    flights = flights[[col for col in flights.columns if col not in columns_to_drop]]
    
    flights[['AIRLINE','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT']] = flights[['AIRLINE','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT']].astype(str)

    flights[['SCHEDULED_DEPARTURE','DEPARTURE_TIME','SCHEDULED_ARRIVAL','ARRIVAL_TIME']]=\
    flights[['SCHEDULED_DEPARTURE','DEPARTURE_TIME','SCHEDULED_ARRIVAL','ARRIVAL_TIME']].applymap(clean_time)
    
    airlines = pd.read_csv('airlines.csv')

    airline_lookup = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()
    
    return flights, airline_lookup


def clean_time(time):
    if pd.isnull(time):
        return np.nan
    elif int(time[:2])==24:
        new_time = dt.time(0,int(time[2:4]))
        return new_time
    else:
        new_time = dt.time(int(time[:2]),int(time[2:4]))
        return new_time


flights, airlines = get_data(nrows = 1000)
