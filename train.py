from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from models.RecurrentAutoencoder import RecurrentAutoencoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
from sklearn.metrics import roc_curve
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from utils.dataset import data
from zoneinfo import ZoneInfo
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
import torch
import copy
import sys
import os
import io

# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
weatherKeys = ['DQR2BTAY7SA53TGCFB2AF3HYV', 'GK75UNTNW66B5UXJLGZRAND2Z']                                                # API Keys of VisualCrossing weather service.

def saveModel(model, scaler, pca, threshold, regressor, netFile="model.pkl"):                                           # Function to save the LSTM-Autoencoder model trained to a file.
    torch.save({"model": model, "scaler":scaler, "pca": pca, "threshold": threshold, "regressor": regressor}, netFile)

def loadModel(device, netFile="model.pkl"):                                                                             # Function to load the LSTM-Autoencoder model trained from a file.
    if os.path.exists(netFile):
        network = torch.load(netFile, weights_only = False, map_location=torch.device(device))
    else:
        sys.exit("No model found.")
    return network['model'], network['scaler'], network['pca'], network['threshold'], network['regressor']

def trasfBatch(rowData, batch_size, size_len, shuffle=False):                                                           # Function that generates a dataset composed of multiple batches, each containing 14 elements, using a sliding window over the dataset, taking one new element at each step.
    dataWithTimestamp = np.array([np.concatenate(([pd.to_datetime(timestamp).timestamp()], values)) for timestamp, values in rowData.items()])
    final_dataset = torch.tensor(np.array([dataWithTimestamp[i:i+size_len] for i in range(dataWithTimestamp.shape[0]-size_len)]), dtype=torch.float64)
    final_dataset = torch.utils.data.TensorDataset(final_dataset.clone().detach())
    return torch.utils.data.DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle)

def readDataset(batch_size, size_len):                                                                                  # Loading the dataset from files and constructing the dataloaders.
    dataset = data(dataset_interpolated_train='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_interpolated_train.csv', dataset_interpolated_test='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_interpolated_test.csv', dataset_duplicated_train='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_duplicated_train.csv', dataset_duplicated_test='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_duplicated_test.csv')
    TrainNoRain, ValNoRain, TestNoRain  = dataset.getNoRainfall()
    TrainRain, _, _, _=dataset.getRainfall()
    TrainRainRegressor, TestRainRegressor, TrainRainTargetRegressor, TestRainTargetRegressor= dataset.getRainfallNDarray()
    tz=data.getTimezone()

    train_loader=trasfBatch(TrainNoRain, batch_size, size_len=size_len, shuffle=True)
    validation_loader=trasfBatch(ValNoRain, batch_size, size_len=size_len, shuffle=False)
    test_loader=trasfBatch(TestNoRain, batch_size, size_len=size_len, shuffle=False)

    anomaly_loader=trasfBatch(TrainRain, batch_size, size_len=size_len, shuffle=False)
    return train_loader, validation_loader, test_loader, anomaly_loader, dataset.getscaler(), dataset.getPCA(), dataset.getNFeatures(), TrainRainRegressor, TestRainRegressor, TrainRainTargetRegressor, TestRainTargetRegressor, tz

def train(model, train_set, val_set, test_set, anomaly_test_set, n_epochs, lr, device, best_loss, patience=10):         # Training function for the LSTM-Autoencoder with early stopping and threshold determination.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.L1Loss(reduction='mean').to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = best_loss
    epochs_no_improve = 0

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        model=model.double()
        train_losses, val_losses, test_losses, anomaly_losses = [], [], [], []

        for (batch,) in train_set:
            optimizer.zero_grad()
            batch= batch[:, :, 1:]                                                                                      # Removing timestamps from each batch.
            seq_true = (batch.to(device))
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for (batch,) in val_set:
                batch= batch[:, :, 1:]                                                                                  # Removing timestamps from each batch.
                seq_true = (batch.to(device))
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

            for (batch,) in test_set:
                batch= batch[:, :, 1:]                                                                                  # Removing timestamps from each batch.
                seq_true = (batch.to(device))
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                test_losses.append(loss.item())

            for (batch,) in anomaly_test_set:
                batch= batch[:, :, 1:]                                                                                  # Removing timestamps from each batch.
                seq_true = (batch.to(device))
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                anomaly_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
    
    model.load_state_dict(best_model_wts)
    return model, history

def decoratorAltWeather(func):
    def wrapper(orario, tz, key, *args, **kwargs):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/catania/{orario}?unitGroup=metric&timezone=GMT&key={key}&include=current"
        data=func(url, *args, **kwargs)
        if data is not None:
            date=datetime.fromtimestamp(data['currentConditions']['datetimeEpoch'], tz=tz).astimezone(ZoneInfo('Europe/Rome')).strftime('%Y-%m-%d %H:%M:%S')
            prec=data['currentConditions']['precip']
            return date, prec
        return datetime.fromtimestamp(orario, tz=tz).strftime('%Y-%m-%d %H:%M:%S'), None
    return wrapper

def decoratorWeather(func):
    def wrapper(orario, tz, key, *args, **kwargs):
        orario = datetime.fromtimestamp(orario, tz=tz).replace(second=0, microsecond=0, minute=0) + timedelta(hours=(datetime.fromtimestamp(orario).minute // 30))
        day=orario.strftime('%Y-%m-%d')
        data=func(f"https://archive-api.open-meteo.com/v1/archive?latitude=37.4922&longitude=15.0704&start_date={day}&end_date={day}&hourly=rain&timezone=GMT", *args, **kwargs)
        if data is not None:
            if None in data['hourly']['rain']:
                data=func(f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude=37.4922&longitude=15.0704&hourly=rain&past_days=5&timezone=GMT&forecast_days=3', *args, **kwargs)
            return orario.strftime('%Y-%m-%d %H:%M:%S'), data['hourly']['rain'][data['hourly']['time'].index(orario.strftime('%Y-%m-%dT%H:%M'))]            
        return orario.strftime('%Y-%m-%d %H:%M:%S'), None
    return wrapper

# @decoratorAltWeather
@decoratorWeather
def makeReq(url):                                                                                                       # Function for API requests.
    response = requests.request("GET", url)
    if response.status_code != 200:
        print('Error on GET method, error code: ', response.status_code)
        return None
    return response.json()

def startTrain(seq_len, device, netFile):                                                                               # Training function for the LSTM-Autoencoder model.
    print("Starting model training...")
    train_loader, validation_loader, test_loader, anomaly_loader, scaler, pca, n_features, regressorTrain, regressorTest, regressorTrainTarget, regressorTestSet, tz=readDataset(batch_size=256, size_len=seq_len)
    model = RecurrentAutoencoder(seq_len, n_features=n_features, embedding_dim=128, device=device)
    model, _=train(model, train_loader, validation_loader, test_loader, anomaly_loader, n_epochs = 500, lr=1e-3, device=device, best_loss = 10000.0)
    regressor=regressorFitting(dataset=regressorTrain, target=regressorTrainTarget, modelLabel='RandomForestRegressor', testSet=regressorTest, targetTestSet=regressorTestSet)
    threshold=calculateROCCurve(model, device, regressor, seq_len)
    saveModel(model, scaler, pca, threshold, regressor, netFile)
    print(f"Model training completed and saved in {netFile}.")

def checkModel(seq_len, device, netFile='model.pkl'):                                                                   # Check for the existence of the model in memory.
    if not os.path.exists(netFile):
        startTrain(seq_len, device, netFile)
    return loadModel(device, netFile)

def useRegressor(regressor, regressor_input):                                                                           # Use of the regressor model.
    return regressor.predict(regressor_input)

def regressorFitting(dataset, target, modelLabel=None, testSet=None, targetTestSet=None):                               # Regressor model fitting.
    target, targetTestSet= target.values.ravel(), targetTestSet.values.ravel()
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=100.0),
        'RandomForestRegressor': RandomForestRegressor(max_depth=30, min_samples_split=10, n_estimators=150, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.2, max_depth=7, n_estimators=150, random_state=42),
        'XGBRegressor': XGBRegressor(colsample_bytree=0.9, learning_rate=0.2, max_depth=7, n_estimators=150, subsample=1, random_state=42),
        'LGBMRegressor': LGBMRegressor(learning_rate=0.2, max_depth=20, n_estimators=150, num_leaves=63, random_state=42, verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(depth=10, iterations=150, l2_leaf_reg=1, learning_rate=0.2, random_state=42, silent=True),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto')
    }

    if (modelLabel is None) or (modelLabel not in models):
        rmseMin, modelMin=10, 'default'
        if testSet is None or targetTestSet is None:
            testSet, targetTestSet= dataset, target
        for name, regressorModel in models.items():
            regressorModel.fit(dataset, target)
            rmse=root_mean_squared_error(targetTestSet, useRegressor(regressorModel, testSet))
            if rmse<rmseMin:
                rmseMin, modelMin, bestRegressor=rmse, name, regressorModel
        # print(f'The best regressor algorithm found is {modelMin}, with a RMSE of {rmseMin:.5f}')
        return bestRegressor
    
    regressor=models[modelLabel]
    regressor.fit(dataset, target)
    return regressor

def testReadDataset(start, end, batch_size, size_len):
    dataset = data(dataset_interpolated_train='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_interpolated_train.csv', dataset_interpolated_test='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_interpolated_test.csv', dataset_duplicated_train='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_duplicated_train.csv', dataset_duplicated_test='/home/elio/COLTRANE-V/hyascope/utils/datasets/dataset_duplicated_test.csv')
    testAlluvione, shouldBe=dataset.takeADate(start, end)
    return trasfBatch(testAlluvione, batch_size, size_len=size_len, shuffle=False), shouldBe, dataset.getTimezone()

def itsRainingAction(timestamp, data, regressor, tz, callAPI=True):                                                     # Function to perform the action in case of rain (should execute the regressor).
    mmRegressor= (useRegressor(regressor, data)).item()
    if callAPI is not False:
        _, mmAPI=makeReq(int((timestamp+timedelta(hours=1)).timestamp()), tz, weatherKeys[0])
    else:
        mmAPI=None
    if mmAPI is not None:
        if mmAPI>mmRegressor or mmAPI==0.0:
            return mmAPI, mmAPI, mmRegressor
    return mmRegressor, mmAPI, mmRegressor
    
def inferenceFromDataset(data, model, device, threshold, regressor, tz, callAPI=True, test=False):                      # Function to perform inferences with the LSTM-Autoencoder network.
    losses, allStatus, mmAPITot, mmRegTot, rain_mm= [], [], {}, {}, {}
    criterion = nn.L1Loss(reduction='mean').to(device)

    with torch.no_grad():
        model = model.eval()
        for (batch,) in tqdm(data) if len(data)>1 else data:
            timestamp = datetime.fromtimestamp(int(batch[:, -1, 0].item()), tz=tz)                                      # Extracting timestamp of the last batch element.
            batch= batch[:, :, 1:]                                                                                      # Removing timestamps from each batch.
            seq_true = (batch.to(device))
            seq_pred = model(seq_true)
            loss=criterion(seq_pred, seq_true)
            losses.append(loss.item())

            rainFlag=1 if loss >= threshold else 0                                                                      # 1-> rainfall forecast detected, 0-> no-rainfall forecast.
            allStatus.append(rainFlag)
            if rainFlag==1:
                rain_mm[timestamp.strftime("%Y-%m-%d %H:%M:%S")], mmAPITot[timestamp.strftime("%Y-%m-%d %H:%M:%S")], mmRegTot[timestamp.strftime("%Y-%m-%d %H:%M:%S")] =itsRainingAction(timestamp, batch[:, -1, :], regressor, tz, callAPI)
            else:
                if test:
                    _, mmAPITot[timestamp.strftime("%Y-%m-%d %H:%M:%S")], _ =itsRainingAction(timestamp, batch[:, -1, :], regressor, tz, callAPI)
                    rain_mm[timestamp.strftime("%Y-%m-%d %H:%M:%S")], _, mmRegTot[timestamp.strftime("%Y-%m-%d %H:%M:%S")] =0.0, 0.0, 0.0
                else:
                    rain_mm[timestamp.strftime("%Y-%m-%d %H:%M:%S")], mmAPITot[timestamp.strftime("%Y-%m-%d %H:%M:%S")], mmRegTot[timestamp.strftime("%Y-%m-%d %H:%M:%S")] =0.0, 0.0, 0.0
    return len(data), allStatus.count(0), mmAPITot, mmRegTot, rain_mm, losses

def plotData(dicts, filenames, titles, figsize=(15, 5), putDate=False):
    if len(dicts) != len(filenames) or len(dicts) != len(titles):
        raise ValueError("Il numero di dizionari deve corrispondere al numero di file di output")
    maxValue = max([max(d.values()) for d in dicts])

    for data, filename, title in zip(dicts, filenames, titles):
        os.makedirs(os.path.dirname(filename)) if not os.path.exists(os.path.dirname(filename)) else None
        fig, ax = plt.subplots(figsize=figsize)

        dates, values = list(data.keys()), list(data.values())
        if all(x is None for x in values):
            print(f'Tutti elementi None in {title}')
            continue

        if putDate:
            asseX = [datetime.strptime(data, '%Y-%m-%d %H:%M:%S').strftime('%H:%M') for data in dates]
            labelX='Ora'
            title=title+f" giorno {datetime.strptime(dates[0], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y')}"
        else:
            asseX= dates
            labelX='Data'

        ax.bar(asseX, values, width=0.5, color='skyblue')
        ax.set_xlabel(labelX)
        ax.set_ylabel('Pioggia (mm)')
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylim(0, maxValue+1)

        buf = io.BytesIO()
        plt.savefig(buf, format='pdf', bbox_inches='tight')
        buf.seek(0)
        
        with open(filename, 'wb') as f:
            f.write(buf.getvalue())
        
        buf.close()
        plt.close(fig)

def changeRegressor(model, scaler, pca, threshold, modelLabel=None, saveFile=True):
    TrainRainRegressor, TestRainRegressor, TrainRainTargetRegressor, TestRainTargetRegressor= (data()).getRainfallNDarray()
    regressor=regressorFitting(dataset=TrainRainRegressor, target=TrainRainTargetRegressor, modelLabel=modelLabel, testSet=TestRainRegressor, targetTestSet=TestRainTargetRegressor)
    saveModel(model, scaler, pca, threshold, regressor, 'model.pkl') if saveFile else None
    return regressor

def doCompleteTest(model, device, threshold, regressor, seq_len):
    train_loader, _, test_loader, anomaly_loader, _, _, _, _, _, _, _, tz=readDataset(batch_size=1, size_len=seq_len)
    total, normalSamples, _, _, _, losses=inferenceFromDataset(train_loader, model, device, threshold, regressor, tz, callAPI=False)
    print(f'Correct train set (not rainy) predictions: {normalSamples}/{total} ({((normalSamples/total)*100):.2f}%)')

    total, normalSamples, _, _, _, _=inferenceFromDataset(test_loader, model, device, threshold, regressor, tz, callAPI=False)
    print(f'Correct test set (not rainy) predictions: {normalSamples}/{total} ({((normalSamples/total)*100):.2f}%)')
    
    total, normalSamples, _, _, _, _=inferenceFromDataset(anomaly_loader, model, device, threshold, regressor, tz, callAPI=False)
    print(f'Correct rainy predictions: {total-normalSamples}/{total} ({(((total-normalSamples)/total)*100):.2f}%)')

def doTargetTest(model, device, threshold, regressor, seq_len, start, end):
    testAlluvione, shouldBe, tzTest=testReadDataset(start=start, end=end, batch_size=1, size_len=seq_len)
    _, _, mmAPITot, mmRegTot, rain_registered, _=inferenceFromDataset(testAlluvione, model, device, threshold, regressor, tzTest, True, test=True)

    shouldBe_commonKeys = {key: shouldBe[key] for key in rain_registered if key in shouldBe}
    plotData([mmAPITot, mmRegTot, rain_registered, shouldBe_commonKeys], ['./test/graphs/apiOpenMeteo.pdf', './test/graphs/regression.pdf', './test/graphs/registered.pdf', './test/graphs/shouldBe.pdf'], ['mm di pioggia registrati da OpenMeteo', 'mm di pioggia registrati dal regressore', 'mm di pioggia registrati', 'mm di pioggia registrati dal SIAS'], putDate=True)

def calculateROCCurve(model, device, regressor, seq_len):
    train_loader, _, test_loader, anomaly_loader, _, _, _, _, _, _, _, tz=readDataset(batch_size=1, size_len=seq_len)
    fakeThreshold=0.0
    _, _, _, _, _, train_losses = inferenceFromDataset(train_loader, model, device, fakeThreshold, regressor, tz, False)
    _, _, _, _, _, test_losses = inferenceFromDataset(test_loader, model, device, fakeThreshold, regressor, tz, False)
    _, _, _, _, _, anomaly_losses = inferenceFromDataset(anomaly_loader, model, device, fakeThreshold, regressor, tz, False)

    fpr, tpr, thresholds = roc_curve(np.concatenate([np.zeros(len(train_losses) + len(test_losses)), np.ones(len(anomaly_losses))]), np.concatenate([train_losses, test_losses, anomaly_losses]))
    return thresholds[np.argmax(tpr - fpr)]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len=14    

    model, scaler, pca, threshold, regressor=checkModel(seq_len=seq_len, device=device)
    print(f"Threshold set to {threshold}")

    # regressor=changeRegressor(model, scaler, pca, threshold, modelLabel='RandomForestRegressor', saveFile=True)

    # doCompleteTest(model, device, threshold, regressor, seq_len)
    # doTargetTest(model, device, threshold, regressor, seq_len, start="2013-02-21 00:00:00", end="2013-02-21 23:00:00")
    doTargetTest(model, device, threshold, regressor, seq_len, start="2023-02-09 17:30:00", end="2023-02-11 00:00:00")
