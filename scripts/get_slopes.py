"""
Created on Jul 2024 
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


'''Gets the data from a CSV file path'''
def get_patient_data(path_src, patient_num):  
    patient_fname = get_patient_csv_filename(path_src, patient_num)
    path_csv = os.path.join(path_src, patient_fname)
    df = pd.read_csv(path_csv)
    return df
  
def get_param_value_dict_for_patient(path_src, patient_num, param_row_num):  
    df = get_patient_data(path_src, patient_num)
    return df.loc[param_row_num][1:]
