"""
Created on Jul 2023
@author: James Manalad 

"""

import sys
from time import process_time
import json
import numpy as np
import gc, os
from datetime import date
import pandas as pd
import sympy as sym
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
import random

'''Gets CSV file path'''
def get_patient_csv_filename(path_src, patient_num):
    for fname in os.listdir(path_src):  
        if fname.split('_')[-1].split('.')[0] == str(patient_num):
            return fname
    print('No csv found for patient: ' + str(patient_num))
    return ''

'''Gets the data from a CSV file path'''
def get_patient_data(path_src, patient_num):  
    patient_fname = get_patient_csv_filename(path_src, patient_num)
    path_csv = os.path.join(path_src, patient_fname)
    df = pd.read_csv(path_csv)
    return df
  
def get_param_value_dict_for_patient(path_src, patient_num, param_row_num):  
    df = get_patient_data(path_src, patient_num)
    return df.loc[param_row_num][1:]
    
'''Gets the domain keys in the CSV file saved format'''
def get_domain_from_keys(keys):
    xvals = []
    for key in keys:
        if 'body' not in key.lower():
            print('Key does not contain the word \'body\': ' + key)
        split_key = key.split('_')[-1].split('-')
        if len(split_key)==1 and 'body' in split_key[0].lower():
            xvals.append(0)
        if len(split_key)>1 and 'body' in split_key[0].lower():
            xvals.append(int(split_key[-1]))
    return np.array(xvals)
    
'''Gets the slope values for a given set of metrics'''
def get_patient_slope_for_num_fx(row_data, num_fx, fx_start=1):
    # print(row_data)
    idx_col = num_fx+1 # +1 because fx 1 has column index 2
    idx_start = fx_start+1
    row_data_trim = row_data[idx_start-1:idx_col+1].dropna() 
        # -1 because the previous column is used to calculate the first slope
        # +1 because the last index specified is excluded
    xvals = get_domain_from_keys(row_data_trim.keys())
    yvals = row_data_trim.values
    assert len(xvals) == len(yvals)
    if len(xvals) > 0:
        best_fit = LinearRegression().fit(xvals.reshape(-1, 1), yvals.reshape(-1, 1))
        slope = best_fit.coef_[0][0]
    else:
        slope = np.NAN
    return slope

'''Gets the domain keys in the CSV file saved format'''
def get_domain_from_keys(keys):
    xvals = []
    for key in keys:
        if 'body' not in key.lower():
            print('Key does not contain the word \'body\': ' + key)
        split_key = key.split('_')[-1].split('-')
        if len(split_key)==1 and 'body' in split_key[0].lower():
            xvals.append(0)
        if len(split_key)>1 and 'body' in split_key[0].lower():
            xvals.append(int(split_key[-1]))
    return np.array(xvals)


def get_paramSlope_df_from_csv_individual(path_src, patient_list, param_name, param_row_num=0, fx_start=1):
    df_param = get_param_df_for_patients(path_src, patient_list, param_name, param_row_num)
    df_slope = get_paramSlope_df_from_param_df(df_param, fx_start)
    return df_slope

# column: body keys
# row: parameter
def get_param_df_for_patients(path_src, patient_list, param_name, param_row_num=0):
    data_dict = {'patient_num':[]}
    body_keys = []
    for patient_num in patient_list:
        dict_patient = get_param_value_dict_for_patient(path_src, patient_num, param_row_num)
        body_keys = list(set(body_keys + list(dict_patient.keys())))
    sorted_body_keys = sort_body_keys(body_keys)
    for body_key in sorted_body_keys:
        key = param_name + '_' + body_key
        data_dict[key] = []
    for patient_num in patient_list:
        data_dict['patient_num'].append(str(patient_num))
        dict_patient = get_param_value_dict_for_patient(path_src, patient_num, param_row_num)
        for body_key in sorted_body_keys:
            key = param_name + '_' + body_key
            # print(body_key)
            if body_key in dict_patient.keys():
                data_dict[key].append(dict_patient[body_key])
            else:
                data_dict[key].append(np.nan)
    df = pd.DataFrame(data_dict)
    return df



def get_paramSlope_df_from_csv_all(path_src_csv, patient_list, fx_start=1):
    df_param = pd.read_csv(path_src_csv)
    df_slope = get_paramSlope_df_from_param_df(df_param, fx_start)
    return df_slope

def get_paramSlope_df_from_param_df(df_param, fx_start=1):
    data_dict_slope = {'patient_num':[]}
    idx_fx_start = fx_start+1
    for column_name in df_param.keys()[idx_fx_start:]:
        key = column_name.split('_')[0]+'-slope_' + '_'.join(column_name.split('_')[1:])
        data_dict_slope[key] = []
    for idx_patient in range(len(df_param.values)):
        row_data = df_param.iloc[idx_patient]
        data_dict_slope['patient_num'].append(row_data['patient_num'])
        for i, key in enumerate(list(data_dict_slope.keys())[1:]): # key at index 0 is patient_num so we start at 1 
            num_fx = fx_start + i
            slope = get_patient_slope_for_num_fx(row_data, num_fx, fx_start)
            data_dict_slope[key].append(slope)
    df_slope = pd.DataFrame(data=data_dict_slope)
    return df_slope
    
