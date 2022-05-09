from torch.utils.data import TensorDataset, DataLoader
import torch
class LSTMData():
    def __init__(self):
        pass
    
    def getTorchLoader(self, X_arr, y_arr, batch_size):
        features = torch.Tensor(X_arr)
        targets = torch.Tensor(y_arr)
        dataSet = TensorDataset(features, targets)
        loader = DataLoader(dataSet, batch_size=batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)
        return dataSet, loader


    def splitDataByRatio(self, data, splitRatio):
        """
        Split Data By Ratio. It usually makes train/validation data and train/test data
        """
        length1=int(len(data)*splitRatio)
        data1, data2 = data[:length1], data[length1:]
        return data1, data2

    def transformXyArr(self, data, transformParameter):
        feature_col= transformParameter["feature_col"]
        target_col= transformParameter["target_col"]
        future_step= transformParameter["future_step"]
        past_step= transformParameter["past_step"]

        self.dataX, self.datay = self._splitXy(data, feature_col, target_col)
        dataX_, datay_ = self._adaptXyByTargetInfo(self.dataX, self.datay, future_step )
        self.dataX_arr, self.datay_arr  = self._getCleanXy(dataX_, datay_, past_step)
        return self.dataX_arr, self.datay_arr

    def _splitXy(self, data, X_col, y_col):
        X = data[X_col]
        y = data[[y_col]]
        return X, y

    def _adaptXyByTargetInfo(self, X, y, future_num, method='step'):
        data_X= X[:-future_num]
        if method=='step':
            if future_num ==0:
                data_y = y
            else:
                data_y = y[future_num:]
        return data_X, data_y

    def _getCleanXy(self, X, y, past_step):
            Clean_X, Clean_y = list(), list()
            Nan_num=0
            print("Original Lenagh:", len(X))
            # Remove set having any nan data
            for i in range(len(X)- past_step+1):
                seq_x = X[i:i+past_step].values
                seq_y = y.iloc[[i+past_step-1]].values
                if np.isnan(seq_x).any() | np.isnan(seq_y).any():
                    Nan_num=Nan_num+1
                else:
                    Clean_X.append(seq_x)
                    Clean_y.append(seq_y)
            print("Removed Data Length:", Nan_num)
            Clean_X = array(Clean_X)
            Clean_y = array(Clean_y).reshape(-1, len(y.columns))
            #Clean_y = array(Clean_y)
            print("Clean Leangth:", len(Clean_X))
            return Clean_X, Clean_y

    
"""
아래 코드 쓰이는가?

"""
import pandas as pd
import numpy as np
from numpy import array
class LearningDataSet():
    def __init__(self, learning_information):
        self.learning_information = learning_information
        self.future_num = learning_information['future_num']
        self.past_num = learning_information['past_num']
        self.target_feature = learning_information['target_feature']
        print("future num:", self.future_num)
    

    def get_LSTMStyle_X(self, data_X):
        print("self.past_num:", self.past_num)
        # if learning method is LSTM
        n_seq = 2
        learning_method = self.learning_information['learning_method']
        n_features = data_X.shape[-1]
        print(n_features)
        #n_features = len(data_X.columns)   
        if learning_method=='CNNLSTM':      
            n_steps = int(self.past_num/n_seq)
            data_X = data_X.reshape((data_X.shape[0],n_seq, n_steps, n_features ))
        elif learning_method =='ConvLSTM':
            # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
            n_steps = int(self.past_num/n_seq)
            data_X = data_X.reshape((data_X.shape[0], n_seq, 1, n_steps, n_features))
        else:
            n_steps = self.past_num

        
        self.learning_information['n_seq'] = n_seq   
        self.learning_information['n_steps'] = n_steps 

        return data_X, self.learning_information
    
    # Separate the original dataset into X and y by the target colum, future num, method
    # multivariate multi-step stacked lstm example   
    # Method: mean, step, others

    def make_dataset_by_target(self, data, method='mean'):
        y = data[[self.target_feature]]
        data_y = pd.DataFrame()
        data_X= data[:len(data)-self.future_num]

        # method == step
        # if future_num is N, data_y(n) is y(n+future_num-1))
        if method=='step':
            data_y = y[(self.future_num-1):]

        #  method == mean
        # if future_num is N, data_y(n) is Mean(y(n)~y(n+(N-1))
        else: 
            for i in range(self.future_num):
                j = i
                y[self.target_feature+'+'+str(j)] = y[self.target_feature].shift(-j)
            y = y.drop(self.target_feature, axis=1)[:len(data)-self.future_num]
            if method=='mean':
                y = y.mean(axis=1)   
            elif method=='max':
                y = y.max(axis=1)
            elif method=='min':
                y = y.min(axis=1)
            else: 
                y = y.mean(axis=1) 
            data_y[self.target_feature+'_CurrentAndFuture_'+method+''+str(self.future_num)] = y
            # Modify the code below to adaptively change the shape of y depending on the situation 
            # by making more specific rules in the future
            
        return data_X, data_y
    
    
   
    


