from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
import joblib
import json
# 2022 New Code

class DataScaler():
    def __init__(self, data, scaling_method, rootPath):
        """
        This class generates a scaler and transforms the data. 
        All information should be described in [rootPath]/scaler_list.json. Before use this class, you can make the empty json file (only describing {})
        Checks whether the scaler file is already saved, and if it exists, it is loaded and used. 
        If it does not exist, a new scaler is created based on the input data and saved .

        The scaler can be set to scale only a limited columns. (use self.setScaleColumns)
        Unless otherwise specified, scalers are created for scalable numeric data.

        Example
        -------
        >>> from KETIPreDataTransformation.general_transformation.dataScaler import DataScaler
        >>> scalerRootpath = os.path.join('/Users','jw_macmini','CLUSTGit','KETIAppMachineLearning','scaler')
        >>> DS = DataScaler(df_features, 'minmax',scalerRootpath )
        >>> DS.setScaleColumns(scaleColumns) # it can be skipped
        >>> result = DS.transform()

        :param data: input data to be scaled
        :type data: dataFrame
        :param scaling_method: scaling method 

        :type scaling_method: string (one of ['minmax','standard','maxabs','robust'])'

        :param rootPath: Root path where the scaler will be stored 
        :type rootPath: String (result of os.path.join('directory1','directory2'....))
        """
        self.scaling_method = scaling_method #
        self.scale_columns = get_scalable_columns(data)
        self.data = data
        self._setScalerInfo(rootPath)
        self.scaler = self._setScaler()

    def _setScalerInfo(self, rootPath):
        """
        This function set scalerListJsonFilePath and update it. and describes detail information in [rootpath]/scaler_list.json
        :param rootPath: Root path where the scaler will be stored 
        :type rootPath: String (result of os.path.join('directory1','directory2'....))

        """
        self.scalerListJsonFilePath = os.path.join(rootPath, "scaler_list.json")
        scaler_list = self.readJson(self.scalerListJsonFilePath)
        encoded_scaler_list = self.encodeHashStyle(self.scale_columns)
        scaler_list[encoded_scaler_list] = self.scale_columns
        self.writeJson(self.scalerListJsonFilePath, scaler_list)
        self.scalerFilePath = os.path.join(rootPath, self.scaling_method, encoded_scaler_list, "scaler.pkl")

    def setScaleColumns(self, scaleColumns):
        """
        The function can be set to scale only a limited columns. (use self.setScaleColumns)
        Unless otherwise specified, scalers are created for scalable numeric data.

        :param scaleColumns: limited column list to be scaled
        :type scaleColumns: string list

        Example
        -------
        >>> from KETIPreDataTransformation.general_transformation.dataScaler import DataScaler
        >>> DS = DataScaler(df_features, 'minmax')
        >>> scaleColumns=['a','b']
        >>> DS.setScaleColumns(scaleColumns) # can skip
        """
        self.scale_columns = scaleColumns
        
        #scaler Manipulation

    def _setScaler(self):
        """
        The function set scaler. (generation or load based on root_path info, scale columns)
        
        Returns: scaler
            scaler
        """
        self.dataToBeScaled = self.data[self.scale_columns]
        if os.path.isfile(self.scalerFilePath):
            scaler = joblib.load(self.scalerFilePath)      
            print("Load scaler File")
        else:
            scaler = self._get_BasicScaler(self.scaling_method) 
            scaler = scaler.fit(self.dataToBeScaled)
            self.save_scaler(self.scalerFilePath, scaler)
            print("Make New scaler File")

        return scaler

    def readJson(self, jsonFilePath):
        """
        The function can read json file.  It can be used to find out column list of scaler file.
        
        Example
        -------
        >>> from KETIPreDataTransformation.general_transformation.dataScaler import DataScaler
        >>> scalerRootpath = os.path.join('/Users','scaler')
        >>> DS = DataScaler(df_features, 'minmax',scalerRootpath )
        >>> scaler = DS.setScaler()
        >>> df_features = DS.transform()
        >>> y = os.path.split(os.path.dirname(DS.scalerFilePath))
        >>> scalerList = DS.readJson(DS.scalerListJsonFilePath)
        >>> scalerList[y[-1]] # print column list of scaler

        Returns: scaler
            scaler
        """
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
        
    def save_scaler(self, scalerFilePath, scaler):
        import os
        dir_name = os.path.dirname(scalerFilePath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        joblib.dump(scaler, scalerFilePath)
        
    def _get_BasicScaler(self, scaler):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

def get_scalable_columns(data):
    integer_columns = list(data.select_dtypes(include=['int64', 'int32']).columns)
    float_columns = list(data.select_dtypes(include=['float64', 'float32']).columns)
    object_columns = list(data.select_dtypes(include=['object']).columns)
    scale_columns = integer_columns + float_columns
    return scale_columns
    
class DataInverseScaler():
    def __init__(self, data, scaling_method, rootPath):
        """
        This class makes inverse scaled data.

        Example
        -------
        >>> from KETIPreDataTransformation.general_transformation.dataScaler import DataScaler
        >>> scalerRootpath = os.path.join('/Users','jw_macmini','CLUSTGit','KETIAppMachineLearning','scaler')
        >>> DIS = DataInverseScaler(df_features, 'minmax',scalerRootpath )
        >>> #DIS.setScaleColumns(scaleColumns) # it can be skipped
        >>> result = DIS.transform()

        :param data: input data to be inverse-scaled
        :type data: dataFrame

        :param scaling_method: scaling method 
        :type scaling_method: string (one of ['minmax','standard','maxabs','robust'])'

        :param rootPath: Root path where the scaler will be stored 
        :type rootPath: String (result of os.path.join('directory1','directory2'....))
        """
        self.scaling_method = scaling_method #
        self.scale_columns = get_scalable_columns(data)
        self.data = data
        self._getScalerFilePath(rootPath)
        self.scaler = self._setScaler()

    def encodeHashStyle(self, text):
        import hashlib
        hash_object = hashlib.md5(str(text).encode('utf-8'))
        hashedText= hash_object.hexdigest()
        return hashedText

    def _getScalerFilePath(self, rootPath):
        """
        This function set scaler file path name
        :param rootPath: Root path where the scaler will be stored 
        :type rootPath: String (result of os.path.join('directory1','directory2'....))

        """
        
        encoded_scaler_list = self.encodeHashStyle(self.scale_columns)
        self.scalerFilePath = os.path.join(rootPath, self.scaling_method, encoded_scaler_list, "scaler.pkl")
        print(self.scalerFilePath)

    def _setScaler(self):
        """
        The function set scaler. (generation or load based on root_path info, scale columns)
        
        Returns: scaler
            scaler
        """
        self.dataToBeScaled = self.data[self.scale_columns]
        if os.path.isfile(self.scalerFilePath):
            scaler = joblib.load(self.scalerFilePath)      
            print("Load scaler File")
        else:
            print("No proper scaler")
            scaler=None

        return scaler
    
    def transform(self):
        """
        The function transform data by inverse-scaler
        Returns: pd.DataFarme
            transformed Data
        """
        inverseScaledData = self.scaler.inverse_transform(self.dataToBeScaled)
        self.inverseScaledData= pd.DataFrame(inverseScaledData, index =self.dataToBeScaled.index, columns =self.dataToBeScaled.columns) 
        return self.inverseScaledData