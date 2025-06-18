from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import itertools
import torch
import sys

def applyPCATraining(data, threshold=0.95):
    features = data
    pca = PCA()
    principal_components = pca.fit_transform(features)

    explained_variance_ratio = pca.explained_variance_ratio_

    cumulative_variance_ratio = explained_variance_ratio.cumsum()
    n_components = len(cumulative_variance_ratio[cumulative_variance_ratio < threshold]) + 1

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)

    newCol = [f"PC{i+1}" for i in range(n_components)]
    pca_data = pd.DataFrame(data=principal_components, columns=newCol)
    return pca_data.values, pca

class data(Dataset):
    def __init__(self, dataset_interpolated_train, dataset_interpolated_test, dataset_duplicated_train, dataset_duplicated_test, target='rainfall_now_agg1h'):
        self.targetFeature=target
        # Carico il dataset di train degli elementi interpolati, senza pioggia
        no_rain_dataset_train = pd.read_csv(dataset_interpolated_train,  index_col='index')
        noRain_dataset_train = no_rain_dataset_train[no_rain_dataset_train[self.targetFeature] == 0.0]                                                             # Extracting without rainfall samples from dataset.
        split_index = int(0.9 * len(noRain_dataset_train))
        noRain_dataset_train, noRain_dataset_val=noRain_dataset_train.iloc[:split_index], noRain_dataset_train.iloc[split_index:]
        targets_noRain_dataset_train= noRain_dataset_train[[self.targetFeature]]
        targets_noRain_dataset_val= noRain_dataset_val[[self.targetFeature]]
        noRainIndexes_train = noRain_dataset_train.index
        noRainIndexes_val = noRain_dataset_val.index
        ndarray_noRain_train = noRain_dataset_train.drop(columns=[self.targetFeature]).astype(np.float64).to_numpy()
        ndarray_noRain_val = noRain_dataset_val.drop(columns=[self.targetFeature]).astype(np.float64).to_numpy()

        self.scaler = StandardScaler()                                                                                                                        # Scaler declaring and fitting.
        self.scaler.fit(ndarray_noRain_train)
        noRain_train=self.scaler.transform(ndarray_noRain_train)
        noRain_val=self.scaler.transform(ndarray_noRain_val)

        noRain_train, self.pca_model=applyPCATraining(noRain_train)                                                                                           # Standardizing without rainfall samples and using PCA model.
        noRain_val=self.pca_model.transform(noRain_val)
        noRain_train_dict=dict(zip(noRainIndexes_train, noRain_train))
        noRain_val_dict=dict(zip(noRainIndexes_val, noRain_val))

        # Carico il dataset di test degli elementi interpolati, senza pioggia
        noRain_dataset_test = pd.read_csv(dataset_interpolated_test,  index_col='index')
        noRain_dataset_test = noRain_dataset_test[noRain_dataset_test[self.targetFeature] == 0.0]                                                                  # Extracting without rainfall samples from dataset.
        targets_rain_dataset_test= noRain_dataset_test[[self.targetFeature]]
        noRainIndexes_test = noRain_dataset_test.index
        ndarray_noRain_test = noRain_dataset_test.drop(columns=[self.targetFeature]).astype(np.float64).to_numpy()
        noRain_test=self.pca_model.transform(self.scaler.transform(ndarray_noRain_test))
        noRain_test_dict=dict(zip(noRainIndexes_test, noRain_test))
        
        # Carico il dataset di train degli elementi duplicati, con pioggia
        rain_dataset_train = pd.read_csv(dataset_duplicated_train,  index_col='index')
        rain_dataset_train = (rain_dataset_train[rain_dataset_train[self.targetFeature] != 0.0])                                                                   # Extracting rainfall samples from dataset.
        targets_rain_dataset_train = rain_dataset_train[[self.targetFeature]]
        rainTrainIndexes = rain_dataset_train.index
        ndarray_train_rain = rain_dataset_train.drop(columns=[self.targetFeature]).astype(np.float64).to_numpy()
        rain_train=self.pca_model.transform(self.scaler.transform(ndarray_train_rain))
        rain_train_dict=dict(zip(rainTrainIndexes, rain_train))

        # Carico il dataset di test degli elementi duplicati, con pioggia
        rain_dataset_test = pd.read_csv(dataset_duplicated_test,  index_col='index')
        rain_dataset_test = (rain_dataset_test[rain_dataset_test[self.targetFeature] != 0.0])                                                                      # Extracting rainfall samples from dataset.
        targets_rain_dataset_test = rain_dataset_test[[self.targetFeature]]
        rainTestIndexes = rain_dataset_test.index
        ndarray_test_rain = rain_dataset_test.drop(columns=[self.targetFeature]).astype(np.float64).to_numpy()
        rain_test=self.pca_model.transform(self.scaler.transform(ndarray_test_rain))
        rain_test_dict=dict(zip(rainTestIndexes, rain_test))

        self.TrainNormalndarray=noRain_train
        self.TestNormalndarray=noRain_test
        self.ValNormalndarray=noRain_val

        self.TrainAnomalyndarray=rain_train
        self.TestAnomalyndarray=rain_test
        self.TrainAnomalyndarrayTarget=targets_rain_dataset_train
        self.TestAnomalyndarrayTarget=targets_rain_dataset_test

        self.TrainNormal=noRain_train_dict
        self.TestNormal=noRain_test_dict
        self.ValNormal=noRain_val_dict

        self.TrainAnomaly=rain_train_dict
        self.TestAnomaly=rain_test_dict

        # import pickle
        # with open('./utils/datasets/dataset_duplicatedOnlyRain.pkl', 'wb') as file:
        #     pickle.dump((self.TrainAnomaly, self.TestAnomaly, self.TrainAnomalyndarray, self.TestAnomalyndarray, self.TrainAnomalyndarrayTarget, self.TestAnomalyndarrayTarget), file)

        if ((self.TrainNormalndarray).shape[1]==(self.TrainAnomalyndarray).shape[1]==(self.TestAnomalyndarray).shape[1]):                                     # Extracting the number of the features.
            self.nFeatures=self.TrainNormalndarray.shape[1]
        else:
            sys.exit("Data length error.")

    def get_torch_tensor(self):
        return torch.from_numpy(self.X)

    def getTestRainfall(self):
        return self.TestAnomaly, self.anomalyTestTarget

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]
    
    def getscaler(self):
        return self.scaler
    
    def getPCA(self):
        return self.pca_model
    
    def getNoRainfall(self):
        return self.TrainNormal, self.ValNormal, self.TestNormal 
    
    def getRainfall(self):
        return self.TrainAnomaly, self.TestAnomaly, self.TrainAnomalyndarrayTarget, self.TestAnomalyndarrayTarget
    
    def getRainfallNDarray(self):
        return self.TrainAnomalyndarray, self.TestAnomalyndarray, self.TrainAnomalyndarrayTarget, self.TestAnomalyndarrayTarget
    
    def getNFeatures(self):
        return self.nFeatures
    
    def getTarget(self):
        return self.targetFeature
    
    def takeADate(self, start, end):
        result, target = {}, {}
        start=datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end=datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        trainRainyTarget = self.TrainAnomalyndarrayTarget.loc[~self.TrainAnomalyndarrayTarget.index.duplicated(keep='first')]
        testRainyTarget = self.TestAnomalyndarrayTarget.loc[~self.TestAnomalyndarrayTarget.index.duplicated(keep='first')]

        for d in [self.TrainAnomaly, self.TestAnomaly, self.TrainNormal, self.ValNormal, self.TestNormal]:
            for k, v in d.items():
                timestamp=datetime.strptime(k, "%Y-%m-%d %H:%M:%S")
                if timestamp >= start and timestamp <= end and k not in result:
                    result[k] = v

                    if k in testRainyTarget.index:
                        target[k] = testRainyTarget.loc[k][self.targetFeature]
                    elif k in trainRainyTarget.index:
                        target[k] = trainRainyTarget.loc[k][self.targetFeature]
                    else:
                        target[k] = 0.0
        return dict(sorted(result.items())), dict(sorted(target.items()))

    def getTimezone(self):
        return timezone.utc
    
    def takeFromFile(self, csvPath):
        testData = pd.read_csv(csvPath,  index_col='index')
        testDataTarget= testData[[self.targetFeature]]
        testDataIndexes = testData.index
        testData = testData.drop(columns=[self.targetFeature]).astype(np.float64).to_numpy()
        testData=self.pca_model.transform(self.scaler.transform(testData))
        testData_dict=dict(zip(testDataIndexes, testData))

        return testData, testDataTarget.squeeze().to_dict(), testData_dict

if __name__ == '__main__':
    dataset = data(dataset_interpolated_train='./utils/datasets/dataset_interpolated_train.csv', dataset_interpolated_test='./utils/datasets/dataset_interpolated_test.csv', dataset_duplicated_train='./utils/datasets/dataset_duplicated_train.csv', dataset_duplicated_test='./utils/datasets/dataset_duplicated_test.csv')
    # print(dataset.takeADate(start="2023-02-10 00:00:00", end="2023-02-10 14:00:00"))
    # print(dataset.takeADate(start="2023-02-09 17:30:00", end="2023-02-11 00:00:00"))

    data, target, dictionary=dataset.takeFromFile('./utils/datasets/ottobre2024/ottobre2024_preprocessed.csv')

    print(data)
    print(target)