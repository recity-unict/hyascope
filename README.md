# hyascope

The project involves forecasting rainfall in millimeters for an hour using machine learning techniques, using the SIAS dataset, which provides data from the Catania, Paternò, and Pedara stations.

# Dependencies

To unzip the 'model.pkl' already trained, the dataset we used for testing and compile SWMM, you can easily run the 'build.sh' file, with the following command:
```bash
  chmod +x ./build.sh
  ./build.sh
```

To provide an environment comparable to ours, you’ll also find the 'env.yml' file — a miniconda configuration file used to create the hyascope-env environment.

To install dependencies, type the following command in the terminal:
```bash
  pip install -r requirements.txt
```

# Algorithm Preprocessing and Training Workflow

Below is the preprocessing procedure (described with code in the respective notebook):
- Merging data from the 3 stations on an hourly basis.
- Obtaining a preliminary dataset by extracting tuples where rainfall is present on an hourly basis. Then, features with a NaN percentage >10% are deleted, followed by the removal of rows containing NaN values.
- A second dataset is obtained by sampling from the original dataset every half hour, followed by linear interpolation to reduce the number of NaN values and obtain meaningful values every 30 minutes. Subsequently, features with more than 2% NaN values are removed, and then all rows with NaN elements are deleted.
- The first dataset is then augmented by duplicating tuples (duplicated based on the percentage of samples without rainfall to balance the final dataset).
- Merging the two datasets.
- Deleting features with more than 2% NaN values (the merge could include features that are absent in one of the datasets, resulting in a high number of NaN values).
- Converting dates into polar coordinates.
- Saving the dataset.

The training procedure is as follows:
- Reading the dataset obtained from preprocessing.
- Removing the target column, which is the prediction of rainfall one hour ahead.
- Standardizing the dataset, followed by PCA application.
- Splitting the dataset into train and test sets.
- Training of the LSTM-Autoencoder network for the classification of rainy or non-rainy events.
- Training the regressor model.
- Saving information regarding the models, the LSTM-Autoencoder threshold (by default the 95 percentile is taken), standardization, and PCA model to a file ('model.pkl').
- Test the models.

# Inference on the Architecture

To perform inferences, a sample service was generated. Here is how the script works:
- The parameters are read, and the existence of the model in memory is checked.
- The arrival of new meteorological data is simulated; in each iteration, the file related to the dataset is read, and the next sample is proposed.
- A window of 'seq_len' meteorological samples (14 by default) will be generated.
- Inference will be performed on the models.
- The result will be saved in a format compatible with SWMM (the generated file will be named in the format 'DDMM.txt', e.g., '10febbraio.txt').

The models input request must be a JSON containing the following fields:

```
{
  date: ['temp_min', 'temp_avg', 'temp_max', 'rainfall_now', 'humidity_min','humidity_avg', 'humidity_max', 'atm_pressure_ist','atm_pressure_ist_rid',
        'wind_speed_2m_avg', 'wind_speed_2m_max', 'wind_dir_2m_', 'wind_speed_10m_avg', 'wind_speed_10m_max', 'wind_dir_10m_avg', 'temp_min_pedara',
        'temp_avg_pedara', 'temp_max_pedara', 'rainfall_now_pedara', 'humidity_min_pedara', 'humidity_avg_pedara', 'humidity_max_pedara', 'wind_speed_2m_avg_pedara',
        'wind_speed_2m_max_pedara', 'wind_dir_2m__pedara', 'temp_min_paternò', 'temp_avg_paternò', 'temp_max_paternò', 'rainfall_now_paternò','humidity_min_paternò',
        'humidity_avg_paternò', 'humidity_max_paternò', 'wind_speed_2m_avg_paternò', 'wind_speed_2m_max_paternò', 'wind_dir_2m__paternò', 'time_sin', 'time_cos']
}
```

Below there is a description of all the features listed above:

- 'temp_min' -> Catania station, original dataset code: 10
- 'temp_avg' -> Catania station, original dataset code: 15
- 'temp_max' -> Catania station, original dataset code: 20
- 'rainfall_now' -> Catania station, original dataset code: 60
- 'humidity_min' -> Catania station, original dataset code: 75
- 'humidity_avg' -> Catania station, original dataset code: 80
- 'humidity_max' -> Catania station, original dataset code: 85
- 'atm_pressure_ist' -> Catania station, original dataset code: 90
- 'atm_pressure_ist_rid' -> Catania station, original dataset code: 95
- 'wind_speed_2m_avg' -> Catania station, original dataset code: 135
- 'wind_speed_2m_max' -> Catania station, original dataset code: 140
- 'wind_dir_2m_' -> Catania station, original dataset code: 155
- 'wind_speed_10m_avg' -> Catania station, original dataset code: 165
- 'wind_speed_10m_max' -> Catania station, original dataset code: 170
- 'wind_dir_10m_avg' -> Catania station, original dataset code: 185
- 'temp_min_pedara' -> Pedara station, original dataset code: 10
- 'temp_avg_pedara' -> Pedara station, original dataset code: 15
- 'temp_max_pedara' -> Pedara station, original dataset code: 20
- 'rainfall_now_pedara' -> Pedara station, original dataset code: 60
- 'humidity_min_pedara' -> Pedara station, original dataset code: 75
- 'humidity_avg_pedara' -> Pedara station, original dataset code: 80
- 'humidity_max_pedara' -> Pedara station, original dataset code: 85
- 'wind_speed_2m_avg_pedara' -> Pedara station, original dataset code: 135
- 'wind_speed_2m_max_pedara' -> Pedara station, original dataset code: 140
- 'wind_dir_2m__pedara' -> Pedara station, original dataset code: 155
- 'temp_min_paternò' -> Paternò station, original dataset code: 10
- 'temp_avg_paternò' -> Paternò station, original dataset code: 15
- 'temp_max_paternò' -> Paternò station, original dataset code: 20
- 'rainfall_now_paternò' -> Paternò station, original dataset code: 60
- 'humidity_min_paternò' -> Paternò station, original dataset code: 75
- 'humidity_avg_paternò' -> Paternò station, original dataset code: 80
- 'humidity_max_paternò' -> Paternò station, original dataset code: 85
- 'wind_speed_2m_avg_paternò' -> Paternò station, original dataset code: 135
- 'wind_speed_2m_max_paternò' -> Paternò station, original dataset code: 140
- 'wind_dir_2m__paternò' -> Paternò station, original dataset code: 155
- 'time_sin' -> Time in polar coordinates;
- 'time_cos' -> Time in polar coordinates.

So here's an example of a sample:
```
{
  '2023-02-10 00:30:00': array([8.47499967e+00, 9.17499983e+00, 9.77499974e+00, 6.00000024e-01, 9.50000000e+01, 9.75000000e+01, 1.00000000e+02, 
                        1.02254999e+03, 1.02383502e+03, 9.34999990e+00, 1.91499996e+01, 5.20000000e+01, 1.35500002e+01, 2.63000002e+01, 
                        5.35000000e+01, 5.00000007e-02, 3.00000012e-01, 5.00000015e-01, 1.00000000e+00, 1.00000000e+02, 1.00000000e+02, 
                        1.00000000e+02, 7.75000000e+00, 1.65500002e+01, 5.70000000e+01, 6.75000000e+00, 7.04999995e+00, 7.45000005e+00, 
                        4.00000006e-01, 8.35000000e+01, 8.55000000e+01, 8.70000000e+01, 6.14999986e+00, 1.39500003e+01, 4.70000000e+01, 
                        8.48755904e-04, 9.99999640e-01])
}
```

note: The LSTM-Autoencoder model expects 'seq_len' samples (by default, this number is set to 14). Therefore, if 'seq_len' samples are not collected first, no inferences will be made with the model. Practically, this means there will be a wait time depending on the 'sampling_time' (the time between one sample and the next) and the number of samples required for inference.

# Running

- To start the data preprocessing phase, type the following command in the terminal:

```bash
  python3 data_processing.py
```

By default, the generated dataset will be stored in the './utils/datasets/' folder.

- To start the model training phase, type the following command in the terminal:

```bash
  python3 train.py
```

An example of a service that performs forecasting is 'service.py'. It requires start/end time for the forecast, sampling time (in minutes), and the dataset from which to retrieve information, in CSV format, as input. The output will include a file named 'DDMM.txt', which is compatible with SWIMM. Additionally, the data recorded in the file will not be mm of rainfall but mm/h (intensity). To execute it, type the following command in the terminal:

```bash
python3 service.py --start "2023-02-10 00:00:00" --end "2023-02-10 16:00:00" --dataset "/path/to/utils/datasets/dataset.csv" --samp_t 30 --INPFile "/path/to/SWMM/swmm_input.INP"
```

Note: you have to replace './utils/datasets/dataset.csv' with the path to the dataset.csv file generated by the 'data_processing.py' script.

# WARNING
The file related to the original dataset ('original.csv') has been zipped due to its size. To use it, unzip the file './utils/datasets/original.csv.zip'.
