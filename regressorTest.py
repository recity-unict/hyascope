from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from utils.dataset import data
from sklearn.svm import SVR
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import torch
import json
import sys

def trasfBatch(rowData, batch_size, size_len, shuffle=False):                               # Function that generates a dataset composed of multiple batches, each containing 14 elements, using a sliding window over the dataset, taking one new element at each step.
    dataWithTimestamp = np.array([np.concatenate(([pd.to_datetime(timestamp).timestamp()], values)) for timestamp, values in rowData.items()])
    final_dataset = torch.tensor(np.array([dataWithTimestamp[i:i+size_len] for i in range(dataWithTimestamp.shape[0]-size_len)]), dtype=torch.float32)
    final_dataset = torch.utils.data.TensorDataset(final_dataset.clone().detach().float())
    return torch.utils.data.DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle)

def readDataset(batch_size, size_len, data_path=None):                                                      # Loading the dataset from files and constructing the dataloaders.
    dataset = data(data_path) if data_path is not None else data()

    TrainNoRain, ValNoRain, TestNoRain  = dataset.getNoRainfall()
    TrainRain, _, _, _=dataset.getRainfall()
    TrainRainRegressor, TestRainRegressor, TrainRainTargetRegressor, TestRainTargetRegressor= dataset.getRainfallNDarray()

    train_loader=trasfBatch(TrainNoRain, batch_size, size_len=size_len, shuffle=True)
    validation_loader=trasfBatch(ValNoRain, batch_size, size_len=size_len, shuffle=False)
    test_loader=trasfBatch(TestNoRain, batch_size, size_len=size_len, shuffle=False)
    anomaly_loader=trasfBatch(TrainRain, batch_size, size_len=size_len, shuffle=False)

    return train_loader, validation_loader, test_loader, anomaly_loader, dataset.getscaler(), dataset.getPCA(), dataset.getNFeatures(), TrainRainRegressor, TestRainRegressor, TrainRainTargetRegressor, TestRainTargetRegressor

def useRegressor(regressor, regressor_input):                                               # Use of the regressor model.
    return regressor.predict(regressor_input)

def regressorFitting(dataset, target, modelLabel=None, testSet=None, targetTestSet=None):   # Regressor model fitting.
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=100.0),
        'RandomForestRegressor': RandomForestRegressor(max_depth=10, min_samples_split=10, n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=100, random_state=42),
        'XGBRegressor': XGBRegressor(colsample_bytree=0.9, learning_rate=0.1, max_depth=3, n_estimators=50, subsample=0.9, random_state=42),
        'LGBMRegressor': LGBMRegressor(learning_rate=0.01, max_depth=20, n_estimators=100, num_leaves=63, random_state=42, verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(depth=6, iterations=50, l2_leaf_reg=3, learning_rate=0.1, random_state=42, silent=True),
        'SVR': SVR(C=10.0, epsilon=0.2, kernel='rbf'),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto')
    }

    if (modelLabel is None) or (modelLabel not in models):
        print(modelLabel)
        rmseMin, modelMin=10, 'default'
        if testSet is None or targetTestSet is None:
            testSet, targetTestSet= dataset, target
        for name, regressorModel in models.items():
            regressorModel.fit(dataset, target)
            rmse=np.sqrt(mean_squared_error(targetTestSet, useRegressor(regressorModel, testSet)))
            print(f'{name}, rmse: {rmse}')
            if rmse<rmseMin:
                rmseMin, modelMin, bestRegressor=rmse, name, regressorModel
        print(f'Il modello migliore è {modelMin}, con un rmse di {rmseMin:.5f}')
        return bestRegressor
    
    regressor=models[modelLabel]
    regressor.fit(dataset, target)
    return regressor

def tryModels(X_train, X_test, y_train, y_test):
    param_grids = {
        'LinearRegression': {},
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBRegressor': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'LGBMRegressor': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127],
            'max_depth': [-1, 10, 20]
        },
        # 'XGBRegressor': {
        #     'n_estimators': [50, 100, 150],
        #     'learning_rate': [0.1, 0.2],
        #     'max_depth': [3, 5, 7],
        #     'subsample': [1.0],
        #     'colsample_bytree': [0.8, 0.9, 1.0]
        # },
        # 'LGBMRegressor': {
        #     'n_estimators': [50, 150],
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'num_leaves': [63, 127],
        #     'max_depth': [-1, 10, 20]
        # },
        'CatBoostRegressor': {
            'iterations': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [6, 8, 10],
            'l2_leaf_reg': [1, 3, 5]
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'XGBRegressor': XGBRegressor(),
        'LGBMRegressor': LGBMRegressor(),
        'CatBoostRegressor': CatBoostRegressor(silent=True),
        'KNeighborsRegressor': KNeighborsRegressor()
    }
    
    results = {}
    for model_name, model in models.items():
        print(f"Performing grid search for {model_name}...")
        param_grid = param_grids[model_name]
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = np.sqrt(-grid_search.best_score_)
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}
            best_score = root_mean_squared_error(y_train, best_model.predict(X_train))
        
        y_pred = best_model.predict(X_test)
        results[model_name] = {
            'Best Parameters': best_params,
            'Best Score (RMSE)/Train RMSE': best_score,
            'Test RMSE': root_mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        print(results[model_name])
    return results

def regressorFitting(models, dataset, testSet, target, targetTestSet):                                                      # Regressor model fitting.
    res={}
    for name, regressorModel in models.items():
        regressorModel.fit(dataset, target)
        res[name]={'Train RMSE': root_mean_squared_error(target, useRegressor(regressorModel, dataset)), 'Test RMSE': root_mean_squared_error(targetTestSet, useRegressor(regressorModel, testSet))}
        print(f'{name} tested, {res[name]}')
    return res

def initModels():
    # file_paths=['./utils/datasets/DatasetsModelloRegressione/dataset_duplicatedAllSamples.pkl', './utils/datasets/DatasetsModelloRegressione/dataset_duplicatedOnlyRain.pkl', './utils/datasets/DatasetsModelloRegressione/dataset_interpolatedAllSamples.pkl', './utils/datasets/DatasetsModelloRegressione/dataset_interpolatedOnlyRain.pkl']
    # Int all samples
    interpolatedAll = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=100.0),
        'RandomForestRegressor': RandomForestRegressor(max_depth=None, min_samples_split=5, n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=50, random_state=42),
        'XGBRegressor': XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=50, subsample=0.8, random_state=42),
        'LGBMRegressor': LGBMRegressor(learning_rate=0.01, max_depth=-1, n_estimators=50, num_leaves=63, random_state=42, verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(depth=6, iterations=100, l2_leaf_reg=5, learning_rate=0.1, random_state=42, silent=True),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto')
    }
    interpolatedAll_path='./utils/datasets/DatasetsModelloRegressione/dataset_interpolatedAllSamples.pkl'

    # Dup all samples
    interpolatedRainy = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=100.0),
        'RandomForestRegressor': RandomForestRegressor(max_depth=30, min_samples_split=2, n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.2, max_depth=7, n_estimators=150, random_state=42),
        'XGBRegressor': XGBRegressor(colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=150, subsample=0.9, random_state=42),
        'LGBMRegressor': LGBMRegressor(learning_rate=0.2, max_depth=-1, n_estimators=150, num_leaves=127, random_state=42, verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(depth=10, iterations=150, l2_leaf_reg=3, learning_rate=0.2, random_state=42, silent=True),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto')
    }
    interpolatedRainy_path='./utils/datasets/DatasetsModelloRegressione/dataset_interpolatedOnlyRain.pkl'

    # Dup rain samples
    duplicatedAll = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=100.0),
        'RandomForestRegressor': RandomForestRegressor(max_depth=20, min_samples_split=10, n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=50, random_state=42),
        'XGBRegressor': XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1, random_state=42),
        'LGBMRegressor': LGBMRegressor(learning_rate=0.1, max_depth=10, n_estimators=50, num_leaves=63, random_state=42, verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(depth=6, iterations=50, l2_leaf_reg=1, learning_rate=0.1, random_state=42, silent=True),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto')
    }
    duplicatedAll_path='./utils/datasets/DatasetsModelloRegressione/dataset_duplicatedAllSamples.pkl'

    # Dup only rain samples
    duplicatedRainy = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=100.0),
        'RandomForestRegressor': RandomForestRegressor(max_depth=None, min_samples_split=10, n_estimators=50, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.2, max_depth=7, n_estimators=150, random_state=42),
        'XGBRegressor': XGBRegressor(colsample_bytree=0.9, learning_rate=0.2, max_depth=7, n_estimators=150, subsample=1, random_state=42),
        'LGBMRegressor': LGBMRegressor(learning_rate=0.2, max_depth=20, n_estimators=150, num_leaves=63, random_state=42, verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(depth=10, iterations=150, l2_leaf_reg=1, learning_rate=0.2, random_state=42, silent=True),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto')
    }
    duplicatedRainy_path='./utils/datasets/DatasetsModelloRegressione/dataset_duplicatedOnlyRain.pkl'

    return {interpolatedAll_path: interpolatedAll,
            interpolatedRainy_path: interpolatedRainy,
            duplicatedAll_path: duplicatedAll,
            duplicatedRainy_path: duplicatedRainy}

def bestRegressorTests():
    models=initModels()

    for path in models.keys():
        None if Path(path).exists() else sys.exit(f'{path} file does not exists.')

    results={}
    for file_path in models.keys():
        print(file_path)
        with open(file_path, 'rb') as file:
            _, _, TrainRainRegressor, TestRainRegressor, regressorTrainTarget, regressorTestTarget = pickle.load(file)
        # results[file_path]=regressorFitting(models, TrainRainRegressor, TestRainRegressor, regressorTrainTarget.values.ravel(), regressorTestTarget.values.ravel())
        results[file_path]=tryModels(TrainRainRegressor, TestRainRegressor, regressorTrainTarget.values.ravel(), regressorTestTarget.values.ravel())

    print(results)
    with open('regressor_results.json', 'w') as file:
        json.dump(results, file)

if __name__ == '__main__':
    # _, _, _, _, _, _, _, regressorTrain, regressorTest, regressorTrainTarget, regressorTestTarget=readDataset(batch_size=256, size_len=14, data_path='./utils/datasets/TEST_DUP/datasetDuplicatoWithTime.csv')

    # # models,predictions = (LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)).fit(regressorTrain, regressorTest, regressorTrainTarget, regressorTestTarget)
    # # print(models, '\n\n',predictions)

    # # print(tryModels(regressorTrain, regressorTest, regressorTrainTarget, regressorTestTarget))
    # regressor=regressorFitting(regressorTrain, regressorTrainTarget, modelLabel=None, testSet=regressorTest, targetTestSet=regressorTestTarget)
    # inputElement = regressorTest[0]
    # pred = (useRegressor(regressor, regressorTest[0].reshape(1, -1))).item()
    # print(pred)
    # # print(np.sqrt(mean_squared_error(regressorTestTarget, useRegressor(regressor, regressorTest))))

    bestRegressorTests()