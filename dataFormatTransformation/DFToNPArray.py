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

    dateList = dfX.index.map(lambda t: t.date()).unique()
    print(dateList)

    for startDate in dateList:
        dfX_partial = dfX[startDate:startDate-timedelta(seconds = 1)]
        dfy_partial = dfy[startDate:startDate-timedelta(seconds = 1)]
        X_partial = dfX_partial.values.transpose()
        y_partial = dfy_partial.values[0][0]
        X.append (X_partial)
        y.append (y_partial)

    X = np.array(X)
    y = np.array(y)
    
    return X, y