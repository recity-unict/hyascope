from train import checkModel, inferenceFromDataset
from datetime import datetime, timedelta, timezone
from utils.dataset import data
import pandas as pd
import numpy as np
import subprocess
import argparse
import torch
import json
import sys
import re
import os

# Example of command to run this file: "python3 service.py --start "AAAA-MM-GG hh:mm:ss" --end "AAAA-MM-GG hh:mm:ss" --dataset "/path/to/dataset.csv" --samp_t 30 --INPfile /path/to/SWMM.inp"

def toJson(INP_FILE, ReportFile, K_LARGE, K_SMALL, LARGE_THRESH, mapID, mapSVGPath, georeference_map, georeference_tubes, outputName):

    INP_FILE = os.path.abspath(INP_FILE)

    nodes, coords, edges, xsections = {}, {}, {}, {}
    section = None
    backdrop_lines = []

    with open(INP_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'): continue
            if line.startswith('['):
                section = line.upper()
                continue
            parts = line.split()
            if section in ('[JUNCTIONS]', '[STORAGE]'):
                nodes[parts[0]] = {'is_outfall': False}
            elif section == '[OUTFALLS]':
                nodes[parts[0]] = {'is_outfall': True}
            elif section == '[COORDINATES]':
                coords[parts[0]] = {'x': float(parts[1]), 'y': float(parts[2])}
            elif section == '[CONDUITS]':
                cid, frm, to = parts[0], parts[1], parts[2]
                edges[cid] = {
                    'name': cid,
                    'inlet': frm,
                    'outlet': to,
                    'section': None,
                    'type': None,
                    'depth': 0.0,
                    'overflowed': False,
                    'vertices': []
                }
            elif section == '[XSECTIONS]':
                cid, shape, geom1 = parts[0], parts[1].lower(), float(parts[2])
                xsections[cid] = {
                    'type_geom': shape,
                    'section': geom1
                }
            elif section == '[VERTICES]':
                cid, x, y = parts[0], float(parts[1]), float(parts[2])
                if cid in edges:
                    edges[cid]['vertices'].append({'x': x, 'y': y})
            elif section == '[BACKDROP]':
                backdrop_lines.append(line)

    # Unisci coordinate nei nodi
    for nid in nodes:
        nodes[nid].update(coords.get(nid, {'x': None, 'y': None}))

    # Completa edges con info da xsections
    for cid, edge in edges.items():
        xsec = xsections.get(cid, {'type_geom': 'unknown', 'section': 0.0})
        section_val = xsec['section']
        edge['section'] = section_val
        edge['type_geom'] = xsec['type_geom']  # temporaneo

    # Completa i vertici includendo inizio e fine
    for eid, e in edges.items():
        start = coords.get(e['inlet'])
        end = coords.get(e['outlet'])

        if not start or not end:
            continue

        verts = e.get("vertices", [])
        if not verts or verts[0] != start:
            verts.insert(0, start)
        if not verts or verts[-1] != end:
            verts.append(end)
        e["vertices"] = verts

    # Parsing RPT
    rpt_data = {}
    inside_links = False
    with open(ReportFile) as f:
        for line in f:
            if 'Link Flow Summary' in line:
                inside_links = True
                continue
            if inside_links and line.strip() == '':
                break
            if inside_links:
                parts = line.split()
                if len(parts) < 7 or parts[0] == 'Link': continue
                name = parts[0]
                flow = float(parts[1])
                depth = float(parts[4])
                rpt_data[name] = {'depth': depth}

    # Finalizza type e overflowed
    for e in edges.values():
        section_val = e['section']
        depth_val = rpt_data.get(e['name'], {}).get('depth', 0.0)
        is_large = section_val > LARGE_THRESH
        e['type'] = 'large' if is_large else 'small'
        e['depth'] = depth_val
        e['overflowed'] = depth_val > (K_LARGE if is_large else K_SMALL)
        e.pop('type_geom', None)

    # Parsing BACKDROP per larghezza, altezza, offset
    width = height = offset_x = offset_y = 0.0
    for line in backdrop_lines:
        parts = re.split(r'\s+', line)
        if parts[0].upper() == "DIMENSIONS" and len(parts) >= 5:
            try:
                x0, y0, x1, y1 = map(float, parts[1:5])
                width = x1 - x0
                height = y1 - y0
                offset_x = x0
                offset_y = y0
            except ValueError:
                print("Errore nel parsing delle DIMENSIONS del BACKDROP")

    json_res = {
        'nodes': nodes,
        'edges': list(edges.values()),
        'map': {
            "width": width,
            "height": height,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "id": mapID,
            "svg": mapSVGPath,
            "georeference_map": georeference_map,
            "georeference_tubes": georeference_tubes
        }
    }

    with open(outputName, "w") as f:
        json.dump(json_res, f, indent=2)

def generateBatch(rowData, batch_size=1):                                                                   # Function that generates a dataset composed of multiple batches, each containing 14 elements, using a sliding window over the dataset, taking one new element at each step.
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset((torch.tensor(np.array([[np.concatenate(([pd.to_datetime(timestamp).timestamp()], values)) for timestamp, values in rowData.items()]]), dtype=torch.float64)).clone().detach()), batch_size=batch_size, shuffle=False)

def sampleComing(file_path, timestamp, target='rainfall_now_agg1h'):
    data = timestamp.replace(second=0, microsecond=0, minute=(0 if timestamp.minute < 30 else 30)).strftime('%Y-%m-%d %H:%M:%S')
    return {data:(pd.read_csv(file_path, index_col='index').drop(columns=[target])).loc[data].values}, timezone.utc

def saveTarget(time, target, fileName='output.csv'): (pd.DataFrame({'date': [time], 'rainfall_mm': [target]})).to_csv(fileName, mode='a', header=False if os.path.isfile(fileName) else True, index=False)

def takeData(time): return f'{time.day} {"gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre"[time.month*8-8:][:8]} {time.year}'

def SWIMMFormat(day, mmForecast, sample_t, fileName):
    with open(fileName, 'w') as file:
        file.write(';EPASWMM Time Series Data - '+takeData(day)+'\n\n')
        for date, target in mmForecast.items():
            file.write((datetime.strptime(date, '%Y-%m-%d %H:%M:%S')).strftime('%m/%d/%Y %H:%M')+'\t'+"{:.3f}".format(target/(sample_t/60.0))+'\n')

def readParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, required=True, help='Starting time (AAAA-MM-GG hh:mm:ss).')
    parser.add_argument('--end', type=str, required=True, help='Ending time (AAAA-MM-GG hh:mm:ss).')
    parser.add_argument('--samp_t', type=int, default=30, help='Sampling time (in minutes).')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path.')
    parser.add_argument('--seq_len', type=int, default=14, help='Data sequence length.')
    parser.add_argument('--INPFile', type=str, required=True, help='SWMM inp file.')
    parser.add_argument('--ReportFile', type=str, help='SWMM rpt file.')
    parser.add_argument('--OutFile', type=str, help='SWMM out file.')
    parser.add_argument('--SWMMfileName', type=str, help='SWMM forecast file name.')
    parser.add_argument('--SWMMupdateTime', type=int, default=3, help='Amount of time after which the SWMM simulation is performed.')
    parser.add_argument('--SWMMPath', type=str, default='./SWMM/swmm5-ubuntu-cli/swmm5', help='SWMM file path.')
    parser.add_argument('--dimMaxWindow', type=int, default=12, help='SWMM maximum samples.')
    parser.add_argument('--SWMMDataJson', type=str, default='./webpage/static/SWMM_data.json', help='SWMM json output file path.')
    args = parser.parse_args()

    if args.start is None or args.dataset is None or args.INPFile is None:
        sys.exit("Inputs missing.")
    if not os.path.exists(args.SWMMPath):
        sys.exit("SWMM file not found.")
    args.SWMMPath=os.path.abspath(args.SWMMPath) if not os.path.isabs(args.SWMMPath) else args.SWMMPath

    args.INPFile = os.path.abspath(args.INPFile)
    if not os.path.isfile(args.INPFile):
        print(f"Errore: il file INP specificato non esiste: {args.INPFile}")
        sys.exit(1)

    args.ReportFile = os.path.abspath(args.ReportFile) if args.ReportFile else os.path.join(os.path.dirname(args.INPFile), "swmm_report.rpt")
    args.OutFile = os.path.abspath(args.OutFile) if args.OutFile else os.path.join(os.path.dirname(args.INPFile), "swmm_output.out")
    args.SWMMfileName = os.path.abspath(args.SWMMfileName) if args.SWMMfileName else os.path.join(os.path.dirname(args.INPFile), "forecastData.txt")

    # args.SWMMfileName = os.path.join(os.path.dirname(args.INPFile), os.path.basename(args.SWMMfileName))
    print(f'SWMM forecast file: {args.OutFile}')
    return datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(args.end, '%Y-%m-%d %H:%M:%S'), args.samp_t, args.dataset, args.seq_len, torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.INPFile, args.ReportFile, args.OutFile, args.SWMMfileName, args.SWMMupdateTime, args.SWMMPath, args.dimMaxWindow, args.SWMMDataJson

def updateSWMMConf(INPfile, start, end, overhead):
    with open(INPfile, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith("START_DATE"):
            lines[i] = f"START_DATE           \t{start.strftime('%m-%d-%Y')}\n"
        if line.startswith("START_TIME"):
            lines[i] = f"START_TIME           \t{start.time()}\n"
        if line.startswith("REPORT_START_DATE"):
            lines[i] = f"REPORT_START_DATE    \t{(start + timedelta(minutes=overhead)).strftime('%m-%d-%Y')}\n"
        if line.startswith("REPORT_START_TIME"):
            lines[i] = f"REPORT_START_TIME    \t{(start + timedelta(minutes=overhead)).time()}\n"
        if line.startswith("END_DATE"):
            lines[i] = f"END_DATE             \t{end.strftime('%m-%d-%Y')}\n"
        if line.startswith("END_TIME"):
            lines[i] = f"END_TIME             \t{end.time()}\n"
    with open(INPfile, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    dataTimestamp, end, samp_t, dataset, seq_len, device, INPfile, ReportFile, OutFile, SWMMfileName, SWMMupdateTime, SWMMPath, dimMaxWindow, SWMM_data_json = readParameters()
    model, scaler, pca, threshold, regressor=checkModel(seq_len=seq_len, device=device)
    print(f"Model threshold set to {threshold}")

    data, window={}, {}
    while end>=dataTimestamp:
        newData, tz=sampleComing(dataset, dataTimestamp)
        dataTimestamp= dataTimestamp+ timedelta(minutes=samp_t)

        date, weatherData = next(iter(newData.items()))
        newData = {date: pca.transform(scaler.transform([weatherData]))[0]}
        data=dict(list((data|newData).items())[-seq_len:])
        if len(data)<seq_len:
            continue
        _, _, _, _, rain_registered, _=inferenceFromDataset(generateBatch(data), model, device, threshold, regressor, tz, callAPI=True)
        window[date]=rain_registered[date]
        dates = sorted(window.keys())
        
        # print(rain_registered)
        if len(window)>dimMaxWindow:
            window.pop(dates[0])
        elif len(window)<2:
            continue

        updateSWMMConf(INPfile, datetime.strptime(dates[0], "%Y-%m-%d %H:%M:%S"), datetime.strptime(dates[-1], "%Y-%m-%d %H:%M:%S"), overhead=0)
        SWIMMFormat(datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), window, samp_t, SWMMfileName)
        result = subprocess.run(f"cd {os.path.dirname(INPfile)} && {SWMMPath} {INPfile} {ReportFile} {OutFile}", shell=True, capture_output=True, text=True)
        print(f"SWMM Output: {result.stdout}")
        toJson(INPfile, ReportFile, 0.7, 0.5, 0.4, "mappa1", "static/img/mappa_new.png", [{"lat":37.52715361723378,"lng":15.06053924560547},{"lat":37.52722168649597,"lng":15.100407600402834},{"lat":37.50448309897232,"lng":15.060667991638185}], [{"lat":37.5267452003563,"lng":15.059251785278322},{"lat":37.52722168649597,"lng":15.102939605712892},{"lat":37.50216800393298,"lng":15.058994293212892}], SWMM_data_json)