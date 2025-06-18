#!/usr/bin/env python
# ruff: noqa: E401, E402, E731

import os

from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return send_from_directory('static','index.html')

@app.route('/georeference_mappa')
def georeference_mappa():
    return send_from_directory('static','georeference_mappa.html')

@app.route('/georeference_tubi')
def georeference_tubi():
    return send_from_directory('static','georeference_tubi.html')


@app.route('/monitor')
def monitor():
    return send_from_directory('static','monitor.html')


if __name__ == "__main__":
    app.run(
        host = os.environ.get('RECITY_HOST', '0.0.0.0'), 
        port = os.environ.get('RECITY_PORT', 80), 
        debug = True,
        use_reloader = True
    )
