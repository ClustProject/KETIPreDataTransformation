from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
import joblib
import json
# 2022 New Code

class DataScaler():
    def __init__(self, data, scaling_method):
        """
        This class generates a scaler and transforms the data. 
        Checks whether the scaler file is already saved, and if it exists, it is loaded and used. 
        If it does not exist, a new scaler is created based on the input data and saved .

        The scaler can be set to scale only a limited columns. (use self.setScaleColumns)
        Unless otherwise specified, scalers are created for scalable numeric data.

        Example
        -------
        >>> from KETIPreDataTransformation.general_transformation.dataScaler import DataScaler
        >>> DS = DataScaler(df_features, 'minmax')
        >>> DS.setScaleColumns(scaleColumns) # can skip
        >>> scalerRootpath = os.path.join('/Users','jw_macmini','CLUSTGit','KETIAppMachineLearning','scaler')
        >>> scaler = DS.setScaler(scalerRootpath)
        >>> result = DS.transform()

        data: pandas.data
            original Input DataFrame
        scaling_method: string
            one of ['minmax','standard','maxabs','robust']
        """ 
        self.scaling_method = scaling_method #
        self.scale_columns = self._get_scalable_columns(data)
        self.data = data

    def setScaleColumns(self, scaleColumns):
        """
        The function can be set to scale only a limited columns. (use self.setScaleColumns)
        Unless otherwise specified, scalers are created for scalable numeric data.

        Example
        -------
        >>> from KETIPreDataTransformation.general_transformation.dataScaler import DataScaler
        >>> DS = DataScaler(df_features, 'minmax')
        >>> scaleColumns=['a','b']
        >>> DS.setScaleColumns(scaleColumns) # can skip
        """
        self.scale_columns = scaleColumns
        #scaler Manipulation

    def setScaler(self, root_path):
        """
        The function set scaler. (generation or load based on root_path info, scale columns)
        root_path: string(os.path.join)
            Root path where the scaler will be stored
        Returns: scaler
            scaler
        """
        jsonFilePath = os.path.join(root_path, "scaler_list.json")
        scaler_list = self.readJson(jsonFilePath)
        encoded_scaler_list = self.encodeHashStyle(self.scale_columns)
        scaler_list[encoded_scaler_list] = self.scale_columns
        self.writeJson(jsonFilePath, scaler_list)

        self.scaleFilePath = os.path.join(root_path, self.scaling_method, encoded_scaler_list)
        self.scalerFileName = os.path.join(self.scaleFilePath, "scaler.pkl")
        self.dataToBeScaled = self.data[self.scale_columns]
        if os.path.isfile(self.scalerFileName):
            self.scaler = self._set_scaler_from_file(self.scalerFileName)        
            print("Load scaler File")
        else:
            scaler = self._get_BasicScaler(self.scaling_method) 
            self.scaler = scaler.fit(self.dataToBeScaled)
            self.save_scaler(self.scalerFileName, self.scaler)
            print("Make New scaler File")

        return self.scaler

    def readJson(self, jsonFilePath):
        with open(jsonFilePath, 'r') as json_file:
            jsonText = json.load(json_file)
        return jsonText

    def writeJson(self, jsonFilePath, text):
        with open(jsonFilePath, 'w') as outfile:
            outfile.write(json.dumps(text))

    def encodeHashStyle(self, text):
        import hashlib
        hash_object = hashlib.md5(str(text).encode('utf-8'))
        hashedText= hash_object.hexdigest()
        return hashedText

    def transform(self):
        """
        The function transform data by scaler
        Returns: pd.DataFarme
            transformed Data
        """
        scaledData = self.scaler.transform(self.dataToBeScaled)
        self.scaledData= pd.DataFrame(scaledData, index =self.dataToBeScaled.index, columns =self.dataToBeScaled.columns)
        return self.scaledData
        
    def save_scaler(self, scaler_file_name, scaler):
        import os
        dir_name = os.path.dirname(scaler_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        joblib.dump(scaler, scaler_file_name)


    def _get_scalable_columns(self, data):
        integer_columns = list(data.select_dtypes(include=['int64', 'int32']).columns)
        float_columns = list(data.select_dtypes(include=['float64', 'float32']).columns)
        object_columns = list(data.select_dtypes(include=['object']).columns)
        scale_columns = integer_columns + float_columns
        return scale_columns
        
    def _get_BasicScaler(self, scaler):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()


        
    def _set_scaler_from_file(self, scaler_file_name):
        self.scaler = joblib.load(scaler_file_name)
        return self.scaler
    
class DataInverseScaler():
    def __init__(self, data, scaling_method):
        """
        This class load scaler and make inverse scaling.
        """
        pass