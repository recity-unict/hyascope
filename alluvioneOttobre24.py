from train import checkModel, trasfBatch, inferenceFromDataset, plotData
import matplotlib.pyplot as plt
from utils.dataset import data
from datetime import datetime
import torch
import json
import os
import io

def doInference(csv_path, jsonfile, seq_len=14, saveJSON=True, batch_size=1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model, _, _, threshold, regressor=checkModel(seq_len=seq_len, device=device)
    print(f"Threshold set to {threshold}")

    dataset=data()
    data_test, target_test, data_test_dict=dataset.takeFromFile(csv_path)

    data_test=trasfBatch(data_test_dict, batch_size=batch_size, size_len=seq_len, shuffle=False)
    _, _, mmAPITot, mmRegTot, rain_registered, _=inferenceFromDataset(data_test, model, device, threshold, regressor, dataset.getTimezone(), True, test=True)

    if saveJSON:
        with open(jsonfile, 'w') as json_file:
            json.dump([csv_path, mmAPITot, mmRegTot, rain_registered, target_test], json_file)
    return [mmAPITot, mmRegTot, rain_registered, target_test]

# def plotData(dicts, filenames, titles, figsize=(15, 5), putDate=False):
#     if len(dicts) != len(filenames) or len(dicts) != len(titles):
#         raise ValueError("Il numero di dizionari deve corrispondere al numero di file di output")
#     maxValue = max([max(d.values()) for d in dicts])

#     for data, filename, title in zip(dicts, filenames, titles):
#         os.makedirs(os.path.dirname(filename)) if not os.path.exists(os.path.dirname(filename)) else None
#         fig, ax = plt.subplots(figsize=figsize)

#         dates, values = list(data.keys()), list(data.values())
#         if all(x is None for x in values):
#             print(f'Tutti elementi None in {title}')
#             continue

#         if putDate:
#             asseX = [datetime.strptime(data, '%Y-%m-%d %H:%M:%S').strftime('%H:%M') for data in dates]
#             labelX='Ora'
#             title=title+f" giorno {datetime.strptime(dates[0], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y')}"
#         else:
#             asseX= dates
#             labelX='Data'

#         ax.bar(asseX, values, width=0.5, color='skyblue')
#         ax.set_xlabel(labelX)
#         ax.set_ylabel('Pioggia (mm)')
#         ax.set_title(title)
#         plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
#         ax.set_ylim(0, maxValue+1)

#         buf = io.BytesIO()
#         plt.savefig(buf, format='pdf', bbox_inches='tight')
#         buf.seek(0)
        
#         with open(filename, 'wb') as f:
#             f.write(buf.getvalue())
        
#         buf.close()
#         plt.close(fig)

def useCommonKeys(dictionaries):
    return [dict(sorted(d.items(), key=lambda item: datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S'))) for d in [{k: d[k] for k in set.intersection(*[set(d.keys()) for d in dictionaries])} for d in dictionaries]]

if __name__ == '__main__':    
    jsonfile='./utils/datasets/ottobre2024/test_data.json'
    csv_path='./utils/datasets/ottobre2024/ottobre2024_preprocessed.csv'

    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as json_file:
            [csvfile, mmAPITot, mmRegTot, rain_registered, target_test] = json.load(json_file)
        dati=[mmAPITot, mmRegTot, rain_registered, target_test] if csvfile == csv_path else doInference(csv_path=csv_path, jsonfile=jsonfile)
    else:
        dati=doInference(csv_path=csv_path, jsonfile=jsonfile)

    common_dati= useCommonKeys(dati)

    ottobre18 = [{key: d[key] for key in d if key.startswith('2024-10-18')}for d in common_dati if isinstance(d, dict)]
    ottobre19 = [{key: d[key] for key in d if key.startswith('2024-10-19')}for d in common_dati if isinstance(d, dict)]
    # ottobre20 = [{key: d[key] for key in d if key.startswith('2024-10-20')}for d in common_dati if isinstance(d, dict)]               # Non contiene piogge

    plotData(ottobre19, ['./utils/datasets/ottobre2024/graphs/apiOpenMeteo.pdf', './utils/datasets/ottobre2024/graphs/regression.pdf', './utils/datasets/ottobre2024/graphs/registered.pdf', './utils/datasets/ottobre2024/graphs/shouldBe.pdf'], ['mm di pioggia registrati da OpenMeteo', 'mm di pioggia registrati dal regressore', 'mm di pioggia registrati', 'mm di pioggia registrati dal SIAS'], putDate=True)
