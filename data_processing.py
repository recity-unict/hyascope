from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import os

def saveNoDuplicato(df, target, datasetRelative_dir, test_size=0.30, saveInDifferentFiles=False):
    df=transformDate(df)
    df.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_complete.csv')), index=True, index_label='index')
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    if saveInDifferentFiles:
        noRainfall_train, rainfall_train= df_train[df_train[target] == 0.0], df_train[df_train[target] != 0.0]
        noRainfall_train.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_train_noRainfall.csv')), index=True, index_label='index')
        rainfall_train.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_train_withRainfall.csv')), index=True, index_label='index')
        
        noRainfall_test, rainfall_test= df_test[df_test[target] == 0.0], df_test[df_test[target] != 0.0]
        noRainfall_test.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_test_noRainfall.csv')), index=True, index_label='index')
        rainfall_test.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_test_withRainfall.csv')), index=True, index_label='index')
    else:
        df_train.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_train.csv')), index=True, index_label='index')
        df_test.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_interpolated_test.csv')), index=True, index_label='index')

def transformDate(df):
    time_of_the_year = (df.index.day_of_year + df.index.hour + df.index.minute) / (365 * 24 * 60)
    df['time_sin'] = np.sin(2 * np.pi * time_of_the_year)
    df['time_cos'] = np.cos(2 * np.pi * time_of_the_year)
    return df

def getRainfallStats(data, feature, printStats=True):
    sunny = (data[feature] == 0.0).mean() * 100
    rainy = (data[feature] != 0.0).mean() * 100
    sunnyDays=(data[feature] == 0.0).sum()
    rainyDays=(data[feature] != 0.0).sum()
    if printStats:
        print("Ci sono:", sunny, '% campioni senza pioggia e ',rainy, '% campioni piovosi.')
        print("Con", sunnyDays, 'campioni senza pioggia e ',rainyDays, 'campioni piovosi.')
    return sunny, sunnyDays, rainy, rainyDays

def getMoreRainy(data, lenTimes):
    return pd.concat([data] * math.ceil((lenTimes*0.4)/len(data)), ignore_index=False)

def shiftRainfall(data, target, freq='1h'):
    data[target] = data[target].shift(freq=freq)
    return data

datasetRelative_dir="./utils/datasets/"
datasetFile=os.path.abspath(os.path.join(datasetRelative_dir, 'original.csv'))
dataset=pd.read_csv(datasetFile)

doTestAlluvione=True
start = "2023-02-09 17:30:00"
end = "2023-02-11 00:00:00"

names = {
    'DATE': 'date',
    'ID_STAZ': 'place',
    '10': 'temp_min',
    '15': 'temp_avg', 
    '20': 'temp_max', 
    '60': 'rainfall_now',
    '75': 'humidity_min',
    '80': 'humidity_avg',
    '85': 'humidity_max',
    '90': 'atm_pressure_ist',
    '92': 'atm_pressure_avg',
    '95': 'atm_pressure_ist_rid',
    '100': 'solar_radiation', 
    '135': 'wind_speed_2m_avg',
    '140': 'wind_speed_2m_max', 
    '155': 'wind_dir_2m_', 
    '165': 'wind_speed_10m_avg', 
    '170': 'wind_speed_10m_max', 
    '185': 'wind_dir_10m_avg'
}

dataset= dataset.rename(columns=names)

dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d%H%M')
dataset['place']=dataset['place'].replace({706:'Catania',712:'Paternò', 713:'Pedara'})

dataset= dataset.dropna(subset=['rainfall_now'])
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.set_index('date', inplace=True)

target='rainfall_now_agg1h'

catania = dataset[dataset["place"].str.contains("Catania")].copy()
catania=catania.drop(columns=['place'])
catania[target] = catania['rainfall_now'].resample('h').sum()
catania=shiftRainfall(catania, target)

pedara=dataset[dataset["place"].str.contains("Pedara")].copy()
pedara[target] = pedara['rainfall_now'].resample('h').sum()
pedara=pedara.drop(columns=['place'])

paterno=dataset[dataset["place"].str.contains("Paternò")].copy()
paterno[target] = paterno['rainfall_now'].resample('h').sum()
paterno=paterno.drop(columns=['place'])

pedaraPaterno = pedara.merge(paterno, on='date', suffixes=('_pedara', '_paternò'))
cataniaMerged = catania.merge(pedaraPaterno, on='date', how='inner')

withoutInterp= cataniaMerged[(cataniaMerged.index.second == 0) & (cataniaMerged.index.minute == 0)]
withoutInterp=withoutInterp.query(target+'> 0.0')
withoutInterp.dropna(axis=1, thresh = int(0.9*withoutInterp.shape[0]), inplace=True)
withoutInterp = withoutInterp.dropna(how='any')

filtered = cataniaMerged[(cataniaMerged.index.second == 0) & ((cataniaMerged.index.minute == 0) | (cataniaMerged.index.minute == 30))]

filtered = filtered.interpolate(method='linear')

filtered.dropna(axis=1, thresh = int(0.98*filtered.shape[0]), inplace=True)
filtered = filtered.dropna(how='any')

# Salvo una copia di 'filtered', che non contiene ancora le duplicazioni. Ovviamente i campioni, in questo caso, mantengono l'ordine temporale.
saveNoDuplicato(filtered, target, datasetRelative_dir)

withoutInterp, test_data = train_test_split(withoutInterp, test_size=0.30, random_state=42)

if doTestAlluvione and start is not None and end is not None:
    test_data = pd.concat([test_data, filtered[(filtered.index >= "2023-02-09 17:30:00") & (filtered.index <= "2023-02-11 00:00:00")]])
else:
    test_data = pd.concat([test_data, filtered])

_, sunnyDays,_,_=getRainfallStats(filtered, target, printStats=False)
withoutInterp=getMoreRainy(withoutInterp, sunnyDays)

filtered=pd.concat([filtered, withoutInterp], axis=0)
filtered.dropna(axis=1, thresh = int(0.98*filtered.shape[0]), inplace=True)
filtered.index = pd.to_datetime(filtered.index)
time_of_the_year = (filtered.index.day_of_year + filtered.index.hour + filtered.index.minute) / (365 * 24 * 60)
filtered['time_sin'] = np.sin(2 * np.pi * time_of_the_year)
filtered['time_cos'] = np.cos(2 * np.pi * time_of_the_year)

time_of_the_year_test = (test_data.index.day_of_year + test_data.index.hour + test_data.index.minute) / (365 * 24 * 60)
test_data['time_sin'] = np.sin(2 * np.pi * time_of_the_year_test)
test_data['time_cos'] = np.cos(2 * np.pi * time_of_the_year_test)

test_data = test_data[filtered.columns.intersection(test_data.columns)]
test_data.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_duplicated_test.csv')), index=True, index_label='index')
filtered.to_csv(os.path.abspath(os.path.join(datasetRelative_dir, 'dataset_duplicated_train.csv')), index=True, index_label='index')

print(filtered.columns)