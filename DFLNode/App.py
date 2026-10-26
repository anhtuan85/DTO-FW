import pandas as pd
from flask_cors import CORS
import os
import pymysql
import json
import requests
from flask import request
from flask import Flask, jsonify, after_this_request
from flask import send_from_directory
from threading import Thread
import importlib
import zipfile
import shutil
import sys

app = Flask(__name__)
CORS(app)   

DTconfig = None
client_config = None


def get_Node_code():
    URL = 'http://' +str(DTconfig['DT_IP'])+':'+str(DTconfig['DT_Port'])+"/SLNode"
    response = requests.get(URL)
    open("SLNode.py", "+wb").write(response.content)
    
def set_configuration():
    global DTconfig
    global client_config
    with open('DT_config.json') as f:
        DTconfig = json.load(f)
    with open('client_config.json') as f:
        client_config = json.load(f)

@app.route('/device_info')
def device_info():
    f = open('device_info.json')
    data = json.load(f)
    return data 

@app.route('/start')
def start():
    param_value = request.args.get('round')
    if(param_value):
        round_num = int(param_value)
    else:
        round_num = 1
    get_Node_code()
    importlib.import_module('SLNode')
    imp_module = importlib.reload(sys.modules['SLNode'])
    node = imp_module.SLNode(client_config['IP'],client_config['Port'], round_num)
    node.start()
    return jsonify({'status': 1})


@app.route('/RL_weights')
def RL_weights():
    param_value = request.args.get('round')
    if(param_value):
        round_num = int(param_value)
    else:
        round_num = 1
    get_Node_code()
    imported_module = importlib.import_module('SLNode')
    imp_module = importlib.reload(sys.modules['SLNode'])
    node = imp_module.SLNode(client_config['IP'],client_config['Port'],round_num)
    node.learn_with_RL()
    return jsonify({'status': 1})

@app.route('/model_to_neighbors',methods=['GET'])
def model_to_neighbors():
    iter_num = request.args.get('iteration')
    path = 'model'+str(iter_num)+'.json.gz'
    root = 'self/'
    return send_from_directory(root, path, as_attachment=True)

# @app.route('/getperformance',methods=['GET'])
# def getperformance():
#     path = 'performance.csv'
#     root = ''
#     return send_from_directory(root, path, as_attachment=True)

@app.route('/getperformance', methods=['GET'])
def getperformance():
    # Directory where the CSV files are located
    directory_to_zip = 'Performance'
    
    # Temporary directory to store the ZIP before sending
    temp_dir = 'temp'
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    
    # Path for the temporary ZIP file
    zip_path = os.path.join(temp_dir, 'performance_files.zip')
    
    # Create a ZIP file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(directory_to_zip):
            for file in files:
                if file.endswith('.csv'):
                    # Create a relative path for files to keep the directory structure
                    rel_dir = os.path.relpath(root, directory_to_zip)
                    rel_file = os.path.join(rel_dir, file)
                    zipf.write(os.path.join(root, file), rel_file)

    # Define a handler to delete the ZIP file after sending it
    @after_this_request
    def remove_file(response):
        try:
            os.remove(zip_path)
            shutil.rmtree(temp_dir)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response
    
    return send_from_directory(temp_dir, 'performance_files.zip', as_attachment=True)

    
if __name__ == '__main__':  
    set_configuration()
    
    # thread = Thread(target=start)
    # thread.start()
    app.run(host='0.0.0.0', port=client_config['Port'])
