
from keras.preprocessing.sequence import TimeseriesGenerator

#train_generator, test_generator, scaler_x, scaler_y, train_x0, train_y0, test_x0, test_y0, train_x, train_y, test_x, test_y= trans_for_learning.get_train_test_lstm_data(station_flag, train, test, target_feature, n_past_steps, n_future_steps)

def get_train_test_lstm_data(station_flag, train, test, target_feature, n_past_steps, n_future_steps):
    train_x = train[:-(n_future_steps-1)]
    train_y = train[[target_feature]].shift(-(n_future_steps-1)).dropna()

    test_x = test[:-(n_future_steps-1)]
    test_y = test[[target_feature]].shift(-(n_future_steps-1)).dropna()
    
    train_x0 = train_x.iloc[0]
    train_y0 = train_y.iloc[n_past_steps-1]
    test_x0 = test_x.iloc[0]
    test_y0 = test_y.iloc[n_past_steps-1]

    scaler_x='no_scaler'
    scaler_y='no_scaler'
    if 'diff' in station_flag:
        train_x, train_y = train_x.diff().fillna(0), train_y.diff().fillna(0)
        test_x, test_y = test_x.diff().fillna(0), test_y.diff().fillna(0)
        
    if 'scale' in station_flag:

        scaler_x, train_x = data_scaling.data_scaling(train_x)
        scaler_y, train_y = data_scaling.data_scaling(train_y)
        test_x = pd.DataFrame(scaler_x.transform(test_x), index=test_x.index, columns=test_x.columns) 
        test_y = pd.DataFrame(scaler_y.transform(test_y), index=test_y.index, columns=test_y.columns) 
    
    if 'log' in station_flag:
        train_x, train_y = np.log(1+train_x), np.log(1+train_y)
        test_x, test_y = np.log(1+test_x), np.log(1+test_y)
    
    train_generator = TimeseriesGenerator(train_x.values, train_y.values, length=n_past_steps, batch_size=1, stride=1)
    test_generator = TimeseriesGenerator(test_x.values, test_y.values, length=n_past_steps, batch_size=1, stride=1)
    
    return train_generator, test_generator, scaler_x, scaler_y, train_x0, train_y0, test_x0, test_y0, train_x, train_y, test_x, test_y