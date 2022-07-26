def transDFtoNP(dfX, dfy):
    """
    Make NumpyArray by input DataFrame.
    
    Return
    - X.shape (sampleNum, featureNum, sequenceNum )
    - y.shape (sampleNum, )

    :param: dfX
    :type: dataFrame
    
    :param: dfy
    :type: dataFrame

    :return: X, y
    :type: numpy array
    
    """
    from datetime import timedelta
    import numpy as np
    start = dfy.index[0].date()
    end = dfy.index[-1].date()
    date = start
    X =[]
    y= []
    while (date <= end) :
        dfX_partial = dfX[dfX.index.date == date]
        dfy_partial = dfy[dfy.index.date == date]
        X_partial = dfX_partial.values.transpose()
        y_partial = dfy_partial.values[0][0]
        X.append (X_partial)
        y.append (y_partial)
        date = date + timedelta(days=1)
    X = np.array(X)
    y = np.array(y)
    
    return X, y