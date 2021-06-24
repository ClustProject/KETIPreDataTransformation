import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd 
import math 

def data_scaling(data, scaler =MinMaxScaler(feature_range=(0, 1))):
         #scaling
        scaler =MinMaxScaler(feature_range=(0, 1))
        data_scale = pd.DataFrame(scaler.fit_transform(data), index=data.index,columns=data.columns)
        return scaler, data_scale