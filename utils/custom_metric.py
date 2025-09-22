import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import json
from utils import decoder

def adjusted(actual_set, pred_set):
    numerator = set(actual_set & pred_set) 


    num = int(len(numerator))
    den = int(len(actual_set))

    try:
        adjusted = float(num / den)
    except:
        if den ==0 and num ==0:
            adjusted = 1
        else:
            adjusted = 0
    return adjusted

def overbooking_similarity(y_true,y_pred, tktno):


    tktno_target = tktno[y_true.index]

    partList = y_true.columns.tolist()

    actual = decoder.ob_pd_decode(y_true,tktno_target,partList)
    pred = decoder.ob_np_decode(y_pred,tktno_target, partList)

    total_adjusted = 0

    for tkt in tktno_target:
        actual_set = actual[tkt]
        pred_set = pred[tkt]

        adjusted_value = adjusted(actual_set, pred_set)
        total_adjusted += adjusted_value
    
    avg_adjusted = total_adjusted/int(len(set(tktno_target)))
    return avg_adjusted