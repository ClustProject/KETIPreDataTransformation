import pandas as pd


# Select only data with "max_nan_num" or less NaN in each repeating clean_duration period.
# If data have dataset_frequency, set frequency finally.

def get_clean_dataset_by_duration(dataset, clean_duration, max_nan_num, dataset_frequency):
    clean_data_set =pd.DataFrame()
    dataset_group_by_time_point = dataset.groupby(pd.Grouper(freq=clean_duration))  
    key_list = dataset_group_by_time_point.groups.keys()
    for key in key_list:
        time_point_data = dataset_group_by_time_point.get_group(key)
        nan_num = time_point_data.isna().sum().sum()
        if nan_num <= max_nan_num:
            clean_data_set= clean_data_set.append(time_point_data)
    if dataset_frequency:
        clean_data_set = clean_data_set.interpolate().asfreq(freq=dataset_frequency)
    return clean_data_set

"""
Example 1)

clean_duration ='D'
max_nan_num =0
dataset_frequency = re_frequency

from KETIPreDataTransformation.trans_for_purpose import cleaning_for_purposes
clean_data_set = cleaning_for_purposes.get_clean_dataset_by_duration(data_set, clean_duration, max_nan_num, dataset_frequency)
"""