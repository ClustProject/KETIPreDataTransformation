
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

        pass
    
    def get_Xy(self, data, target_data_preparation_method):
                                                
        n_features = len(data.columns)                                 
        data_X_df, data_y_df = self.make_dataset_by_target(data, target_data_preparation_method)
        
        # if learning method is LSTM
        learning_information = self.learning_information
        learning_parameter = learning_information['learning_parameter']
        learning_method = learning_information['learning_method']
        

        print("self.past_num:", self.past_num)
        data_X, data_y = self.make_dataset_for_LSTM_style(data_X_df, data_y_df, self.past_num)
        print(data_X.shape)
        n_seq = 2
        if learning_method=='CNNLSTM':      
            n_steps = int(self.past_num/n_seq)
            data_X = data_X.reshape((data_X.shape[0],n_seq, n_steps, n_features ))
        elif learning_method =='ConvLSTM':
            # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
            n_steps = int(self.past_num/n_seq)
            data_X = data_X.reshape((data_X.shape[0], n_seq, 1, n_steps, n_features))
        else:
            n_steps = self.past_num
        learning_information['n_seq'] = n_seq   
        learning_information['n_steps'] = n_steps 

        return data_X, data_y, learning_information
    
    # Separate the original dataset into X and y by the target colum, future num, method
    # Method: mean, step, others
    
    def get_inference_X(self, data):
                                                
        n_features = len(data.columns)                                 
        data_X_df = data
        
        # if learning method is LSTM
        learning_information = self.learning_information
        learning_style = learning_information['learning_style']
        learning_method = learning_information['learning_method']
        
        if learning_style == 'LSTM':
            data_X = data_X_df.values.reshape(-1, self.past_num, n_features)
            print(data_X.shape)
            n_seq = 2
            if learning_method=='CNNLSTM':      
                n_steps = int(self.past_num/n_seq)
                redata_X = data_X.reshape((data_X.shape[0],n_seq, n_steps, n_features ))
            elif learning_method =='ConvLSTM':
                # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
                n_steps = int(self.past_num/n_seq)
                data_X = data_X.reshape((data_X.shape[0], n_seq, 1, n_steps, n_features))
            else:
                n_steps = self.past_num
            learning_information['n_seq'] = n_seq   
            learning_information['n_steps'] = n_steps 
        
        #Modify code below to adaptively for your learning purposes
        else:
            data_X = data_X_df.values.reshape(-1, self.past_num, n_features)
        return data_X, learning_information
    
    # Separate the original dataset into X and y by the target colum, future num, method
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
    
    
    # multivariate multi-step stacked lstm example   
    def make_dataset_for_LSTM_style(self, X, y, n_steps):
        data_X, data_y = list(), list()
        Nan_num=0
        
        # Remove set having any nan data
        for i in range(len(X)- n_steps):
            seq_x = X[i:i+n_steps].values
            seq_y = y.iloc[[i+n_steps]].values
            if np.isnan(seq_x).any() | np.isnan(seq_y).any():
                Nan_num=Nan_num+1
            else:
                data_X.append(seq_x)
                data_y.append(seq_y)
        print("Removed Data Length:", Nan_num)
        data_X = array(data_X)
        data_y = array(data_y).reshape(-1)

        return data_X, data_y


