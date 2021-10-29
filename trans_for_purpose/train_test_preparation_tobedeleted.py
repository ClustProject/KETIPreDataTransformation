import sys
sys.path.append("../")
sys.path.append("../..")

import pandas as pd
import numpy as np
from KETIPreDataTransformation.general_transformation import data_scaling
#inverse transformation


    
#General
def get_train_test_data(station_flag, data, test_length):
    scaler='no_scaler'
    X_train, X_test = data[:-test_length], data[-test_length:]
    X_train_0, X_test_0= X_train.iloc[0], X_test.iloc[0]
    if 'diff' in station_flag:
        X_train, X_test = X_train.diff().fillna(0), X_test.diff().fillna(0)
    if 'scale' in station_flag:
        scaler, X_train = data_scaling.data_scaling(X_train)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns) 
    return X_train, X_train_0, X_test, X_test_0, scaler


def diff_scale_inverse_transform(offset, pred, scaler, station_flag):
    if 'log' in station_flag:
        pred = np.exp(pred)
        pred = pred-1
        
    if 'scale' in station_flag:
        pred = scaler.inverse_transform(pred)
        
    if 'diff' in station_flag:
        pred = offset + pred.cumsum()
    
    return pred