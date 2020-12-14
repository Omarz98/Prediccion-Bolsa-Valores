# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:15:06 2020

@author: 86453
"""


from alpha_vantage.timeseries import TimeSeries
import json

def save_dataset(symbol):
    api_key ='C2KG22FO4I0U600O'
    
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')
    data.to_csv('./{}_daily.csv'.format(symbol))
    
simbolo = 'AAPL'
save_dataset(simbolo)