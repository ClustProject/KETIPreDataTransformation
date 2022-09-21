import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")

def getSplitAndTransformDataByFrequency(data, splitNum, splitInterval, transformFreqList, freqTransformMode):
    """
    Create divided data according to the input number of splits and transform each data according to the entered time frequency.
    Transformation methods according to time frequency are deletion and averaging sampling methods.
    
    :param data: DataFrame with time stamp as index
    :type data: 2D DataFrame
    
    :param splitNum: number of split data
    :type splitNum: Integer
    
    :param splitInterval: split data interval
    :type splitInterval: Interger
    
    :param transformFreqList: List of transform time frequency for each data
    :type: List of integers
    
    :param  freqTransformMode: Transformation methods according to time frequency
    :type: String
    
    :return dataset: split dataset
    :rtype: Dict 
    """
    columns = data.columns
    dataset = {}
    start_interval = 0
    for num in range(splitNum):
        data_c = data.copy()
        ## 서로 다른 주기 별 데이터 생성
        if freqTransformMode == "drop":
            ## data frequency transform
            trans_data = data_c.resample(transformFreqList[num]).first()
        else: # freqTransformMode == "sampling"
            trans_data = data_c.resample(transformFreqList[num]).mean()

        print("split num : ", num)
        print("split data shape : ", trans_data.shape)
        print("------")
        
        if splitNum == 1:
            dataset = trans_data.copy()
        ## get split data
        else:
            end_interval = start_interval+splitInterval[num]
            split_data = trans_data[columns[start_interval:end_interval]]
            start_interval = end_interval
            ## get split data set
            dataset[num] = split_data

    return dataset