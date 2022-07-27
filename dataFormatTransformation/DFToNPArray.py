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
    import datetime as dt
    import numpy as np
    start = dfy.index[0].date()
    end = dfy.index[-1].date()
    date = start
    X =[]
    y= []

    dateList = dfX.index.map(lambda t: t.date()).unique()
    print(dateList)

    for startDate in dateList:
        endDate  = dt.datetime.combine(startDate, dt.time(23, 59, 59, 59))
        dfX_partial = dfX[startDate:endDate]
        dfy_partial = dfy[startDate:endDate]
        X_partial = dfX_partial.values.transpose()
        y_partial = dfy_partial.values[0][0]
        X.append (X_partial)
        y.append (y_partial)

    X = np.array(X)
    y = np.array(y)
    
    return X, y