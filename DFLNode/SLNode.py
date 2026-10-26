# -*- coding: utf-8 -*-
"""
On DT File
"""
import time
import uuid
import pandas as pd
import os
import pymysql
import numpy as np
np.object = object
np.bool = bool 
np.int = int    
np.float = float  
import requests
import keras
from tensorflow.keras.models import model_from_json
from  tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import json
import compress_json
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
# import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from itertools import product
from sklearn.metrics import accuracy_score, precision_score

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DBManager():
    def __init__(self):
        self.con = self.getDBConnection()
        
    def query_executer(self,query):
        try:
            cur = self.con.cursor()
            query = query
            cur.execute(query)
            self.con.commit()
        except Exception as e:
            print('Connection failed -- Insert-- Trying Again', str(e))
            # SLNode().update_log("--DB Connection Error-INSERT--",8)
            self.query_executer(query)
        
    def get_dt_configuration(self):
        with open('DT_config.json') as f:
            DTconfig = json.load(f)
        return DTconfig
    
    def getDBConnection(self):
        DTconfig = self.get_dt_configuration()
        conn = pymysql.connect(
            host = DTconfig['DB_IP'],
            port = int(DTconfig['DB_port']),
            user = DTconfig['DB_username'],
            passwd = DTconfig['DB_pass'],
            db = DTconfig['DB_name'],
            connect_timeout=100,
            charset = 'utf8mb4')
        return conn
    
    def select_query(self,query):
        try:
            self.con = self.getDBConnection()
            result = pd.read_sql_query(query, self.con)
        except Exception as e:
            print('Connection failed -- Select-- Trying Again', str(e))
            # SLNode().update_log("--DB Connection Error-SELECT--",8)
            self.select_query(query)
        return result
    
class SLNode():
    def __init__(self, ip,port, roundnum):
        self.ip = ip
        self.port = port
        self.DB = DBManager()
        self.problem_type = None
        self.DTConfig = self.DB.get_dt_configuration()
        self.id = self.get_device_id()
        self.DTURL = 'http://'+self.DTConfig['DT_IP']+':'+str(self.DTConfig['DT_Port'])+'/'
        config = self.get_configurations()
        self.rounds = config['rounds'][0]
        self.epochs = config['epochs'][0]
        self.model_id = config['m_id'][0]
        self.model = self.load_model()
        self.dataset = config['dataset'][0]
        self.current_round = roundnum
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()
        self.neighbors = self.get_neighbors()
        self.initilize_dir_and_files()
        self.early_stop = False
        self.waiting_time = {}
        self.loss_on_neighbor_models = {}
        self.ErrorMethods = ErrorMethods(self.X_train, self.X_test, self.y_train, self.y_test)
        self.first_iteration = True
        if(roundnum != 1):
            self.initilize_for_between_start()
        
    def initilize_dir_and_files(self):
        Path('self').mkdir(parents=True, exist_ok=True)
        Path('neighbors').mkdir(parents=True, exist_ok=True)
        Path('waiting time').mkdir(parents=True, exist_ok=True)
        Path('Performance').mkdir(parents=True, exist_ok=True)
    
    def get_device_id(self):
        device_id = self.DB.select_query("select id from devices where device_ip ='"+str(self.ip)+"' and port ="+str(self.port))
        device_id = device_id.values.tolist()
        return device_id[0][0]
    
    def initilize_for_between_start(self):
        q = 'select * from rl_weights  where n_to = '+str(self.id) + ';'
        df = self.DB.select_query(q)
        for index, row in df.iterrows():
            try:
                self.waiting_time[str(row['n_from'])].append(float(row['waiting_time']))
            except:
                self.waiting_time[str(row['n_from'])] = [float(row['waiting_time'])]
            
            try:
                self.loss_on_neighbor_models[str(row['n_from'])].append(float(row['loss']))
            except:
                self.loss_on_neighbor_models[str(row['n_from'])] = [float(row['loss'])]
        
        
    def old_prepare_Energy_data(self):
        self.problem_type = 'Regression'
        df = pd.read_csv(self.dataset+'.csv')
        #drop = ['Building','Appartment','Date']
        drop = ['Building','Date','Unnamed: 0']
        X = df.drop(drop, axis = 1)
        cols = X.columns
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X),columns=cols) #['Temperature','Humidity','Day of Week','Day of Month', 'Month', 'Weekend','Energy']
        y = df['Energy']
        X = X.drop('Energy', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        #For LSTMCNN
        #X_train=np.array(X_train).reshape(X_train.shape[0],X_train.shape[1],-1)
        self.update_log("Preparing Energy Data",8)
        return X_train, X_test, y_train, y_test
    

    def prepare_Energy_data(self):
        self.problem_type = 'Regression'
        df = pd.read_csv(self.dataset+'.csv')
        #drop = ['Building','Appartment','Date']
        drop = ['Building','Date','Unnamed: 0']
        X = df.drop(drop, axis = 1)
        cols = X.columns
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X),columns=cols) #['Temperature','Humidity','Day of Week','Day of Month', 'Month', 'Weekend','Energy']
        y = df['Energy']
        X = X.drop('Energy', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        #For LSTMCNN
        #X_train=np.array(X_train).reshape(X_train.shape[0],X_train.shape[1],-1)
        self.update_log("Preparing Energy Data",8)
        return X_train, X_test, y_train, y_test
    

    def prepare_CIFAR10_data(self):
        self.problem_type = 'Classification'
        x_train = np.load("Non-IID-FEMNIST/x_train.npy")
        y_train = np.load("Non-IID-FEMNIST/y_train.npy")
        x_test = np.load("Non-IID-FEMNIST/x_test.npy")
        y_test = np.load("Non-IID-FEMNIST/y_test.npy")
        self.update_log("Preparing CIFAR-10 Data",8)

        # global_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # num_classes = len(global_classes)
        # class_to_index = {label: index for index, label in enumerate(global_classes)}
        # y_train = self.consistent_one_hot(y_train, class_to_index, num_classes)
        # y_test = self.consistent_one_hot(y_test, class_to_index, num_classes)
        
        return x_train, x_test, y_train, y_test

    
    # def prepare_CIFAR10_data(self):
    #     self.problem_type = 'Classification'
    #     df = pd.read_csv(self.dataset+'.csv')
    #     y = df['label']
    #     X = df.drop(['label'], axis = 1)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #     return X_train, X_test, y_train, y_test
    

    
    def prepare_CIFAR10Img_data(self):
        self.problem_type = 'Classification'
        x_train = np.load('CIFARnonIID(C_owntest)/x_train.npy')
        y_train = np.load('CIFARnonIID(C_owntest)/y_train.npy')
        x_test = np.load("CIFARnonIID(C_owntest)/x_test.npy")
        y_test = np.load("CIFARnonIID(C_owntest)/y_test.npy")
        self.update_log("Preparing CIFAR10 Data",8)
        x_train = x_train / 255.0  # Normalize the pixel values to [0, 1]
        x_test = x_test / 255.0
        
        global_classes = [0, 1, 2, 3, 4, 5, 6, 7,8, 9]
        num_classes = len(global_classes)
        class_to_index = {label: index for index, label in enumerate(global_classes)}
        try:
            y_train = [i[0] for i in y_train]
            y_test = [i[0] for i in y_test]
        except:
            print(' -- No Sub Indexes -- ')
        y_train = self.consistent_one_hot(y_train, class_to_index, num_classes)
        y_test = self.consistent_one_hot(y_test, class_to_index, num_classes)
        
        # Split into training and testing sets
       # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test
        
        # (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        # train_images, test_images = train_images / 255.0, test_images / 255.0
    
    def prepare_MINIST_data(self):
        self.problem_type = 'Classification'
        x_train = np.load("Non-IID-FashionMNIST/x_train.npy")
        y_train = np.load("Non-IID-FashionMNIST/y_train.npy")
        x_test = np.load("Non-IID-FashionMNIST/x_test.npy")
        y_test = np.load("Non-IID-FashionMNIST/y_test.npy")
        # x_train = x_train / 255.0
        # x_test = x_test / 255.0
        self.update_log("Preparing MNIST Data",8)

        # global_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # num_classes = len(global_classes)
        # class_to_index = {label: index for index, label in enumerate(global_classes)}
        # y_train = self.consistent_one_hot(y_train, class_to_index, num_classes)
        # y_test = self.consistent_one_hot(y_test, class_to_index, num_classes)
        
        return x_train, x_test, y_train, y_test
    
    
    
    def prepare_Thermal_data(self):
        self.problem_type = 'Classification'
       # df = pd.read_csv("D:\\Projects\Federated Learning\\Datasets\\Fed Dataset Thermal (Transfer Learning Paper)\\"+self.country+".csv")
        df = pd.read_csv('Thermal.csv')
        y = df['Thermal sensation']
        drop_list = ['Unnamed: 0','Publication (Citation)','Data contributor','Heating strategy_building level',
                     'Year','Koppen climate classification','Climate','Building type','Database',
                     'City','Country','Outdoor monthly air temperature (¡C)','Thermal preference','Season','Cooling startegy_building level',
                     'Air movement preference','Humidity preference','Thermal comfort','Thermal sensation','Clo','label']
        drop = list(set(df.columns).intersection(drop_list))
        X = df.drop(drop, axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_train=to_categorical(y_train,num_classes=5)
        y_test=to_categorical(y_test,num_classes=5)
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        y_train = np.reshape(y_train, (len(y_train), len(y_train[0])))
        return X_train, X_test, y_train, y_test
    
    def prepare_CIFAR10pt_data(self):
        # self.problem_type = 'Classification'
        # data = torch.load('CIFAR10 pt/train.pt')
        # x_train = [t[0] for t in data]
        # x_train = np.asarray(x_train).astype(float)
        # y_train = [t[1] for t in data]
        # y_train = np.asarray(y_train).astype(float)
        
        # xy_test = torch.load('CIFAR10 pt/test.pt')
        # x_test = [t[0] for t in xy_test]
        # x_test = np.asarray(x_test).astype(float)
        
        # y_test = [t[1] for t in xy_test]
        # y_test = np.asarray(y_test).astype(float)

        return x_train, x_test, y_train, y_test
    
    def consistent_one_hot(self, labels, class_to_index_map, num_classes):
        label_indices = np.array([class_to_index_map[label] for label in labels])
        encoded_labels = to_categorical(label_indices, num_classes=num_classes)
        return encoded_labels
    
    def prepare_Synthetic_data(self):
        self.problem_type = 'Classification'
        df = pd.read_csv(self.dataset+'.csv')
        y = df['label']
        X = df.drop(['label'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # y_train=to_categorical(y_train,num_classes=5)
        # y_test=to_categorical(y_test,num_classes=5)
        global_classes = [0, 1, 2, 3, 4]
        num_classes = len(global_classes)
        class_to_index = {label: index for index, label in enumerate(global_classes)}
        y_train = self.consistent_one_hot(y_train, class_to_index, num_classes)
        y_test = self.consistent_one_hot(y_test, class_to_index, num_classes)
        
        self.update_log("Preparing Synthetic Data",8)
        return X_train, X_test, y_train, y_test
    
    def prepare_Covtype_data(self):
        self.problem_type = 'Classification'
        df = pd.read_csv(self.dataset+'.csv')
        y = df['label']
        X = df.drop(['label'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # y_train=to_categorical(y_train,num_classes=5)
        # y_test=to_categorical(y_test,num_classes=5)
        global_classes = [1, 2, 3, 4, 5, 6, 7]
        num_classes = len(global_classes)
        class_to_index = {label: index for index, label in enumerate(global_classes)}
        y_train = self.consistent_one_hot(y_train, class_to_index, num_classes)
        y_test = self.consistent_one_hot(y_test, class_to_index, num_classes)
        
        self.update_log("Preparing Covtype Data",8)
        return X_train, X_test, y_train, y_test
    
    def prepare_Churn_data(self):
        self.problem_type = 'Classification'
        df = pd.read_csv(self.dataset+'.csv')
        y = df['churn']
        X = df.drop(['recordID (Source: Customer Signature for Churn Analysis)', 'state', 'account_length', 'churn', 'area_code', 'voice_mail_plan', 'customer_id','international_plan'], axis = 1)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def load_data(self):
        prepare_data = "self.prepare_"+self.dataset+"_data"
        X_train, X_test, y_train, y_test = eval(prepare_data)()
        return X_train, X_test, y_train, y_test
    
    def get_configurations(self):
        con = self.DB.getDBConnection()
        config = pd.read_sql_query("select * from configuration order by conf_id desc limit 1;", con)
        return config.to_dict()
       
    def get_neighbors(self):
        neighbors = self.DB.select_query('select * from topology ,devices where topology.end_node = '+str(self.id)+' and topology.start_node = devices.id')
        df = neighbors.loc[:, ["start_node","device_ip","port"]]
        df = df.values.tolist()
        return df
    
    def get_initial_model(self):
        model_info = self.DB.select_query("select * from models where id = "+str(self.model_id)+";")
        model_json_text = model_info['model_json'][0]
        model_loss = model_info['m_loss'][0]
        model_optimizer = model_info['m_optimizer'][0]
        self.model_loss = model_loss
        model_json_text = model_json_text[1:-1]
        json_model = json.loads(model_json_text)
        json_model = json.dumps(json_model)
        model = model_from_json(json_model)
        
        if(self.problem_type == 'Classification'): 
            pass
            # Set metrics here for Classification or regreaion problems
        
        model.compile(loss = model_loss, optimizer = model_optimizer,metrics=['accuracy'])
        print(model.summary())
        return model
        # URL = self.DTURL+"initial_model_trasfer"
        # response = requests.get(URL)
        # open("model", "+wb").write(response.content)
    
    def load_model(self):
        model = self.get_initial_model()
        # model = keras.models.load_model('model')
        return model

    def train_model(self):
        self.update_log('Training Started', 2)
        history = self.model.fit(self.X_train,self.y_train, epochs=self.epochs, validation_data=(self.X_test,self.y_test)) # , verbose=0
        print('** Local Training Completed **')
        self.update_log('Training Completed', 3)
        performance = history.history
        self.save_model_weights(performance)
        # self.save_to_edgeX(performance)
        self.save_history(performance)
        self.test_classification_model_for_precision('before')

    def make_event_edgex(self, string_data, deviceName, profileName, sourceName):
        event_dict = dict()
        time_it = time.time()
        event_dict["apiVersion"] = "v2"
        event_dict["event"] = dict()
        event_dict["event"]["apiVersion"] = "v2"
        event_dict["event"]["deviceName"] = deviceName
        event_dict["event"]["profileName"] = profileName
        event_dict["event"]["sourceName"] = sourceName
        event_dict["event"]["id"] = str(uuid.uuid1())
        event_dict["event"]["origin"] = int(time_it)
        event_dict["event"]["readings"] = []
        list_data = dict()
        list_data["deviceName"] = deviceName
        list_data["profileName"] = profileName
        list_data["resourceName"] = "data"
        list_data["id"] = str(uuid.uuid1())
        list_data["origin"] = int(time_it)
        list_data["valueType"] = "String"
        list_data["value"] = str(string_data)
        event_dict["event"]["readings"].append(list_data)
        event_json = json.dumps(event_dict)
        return event_json

    def save_to_edgeX(self, performance):
        weights = self.model.get_weights()
        payload = {'weights': weights}
        payloads = json.dumps(payload, cls=NumpyEncoder)
        deviceName = "Node"
        profileName = "DFLNode"
        sourceName = "Models"
        headers = {'Content-Type': 'application/json'}
        event_url = "http://" + self.ip + ":59880/api/v2/event"

        url_request = event_url + "/" + profileName + "/" + deviceName + "/" + sourceName
        event_json = self.make_event_edgex(payloads, deviceName, profileName, sourceName)
        res = requests.post(url = url_request, data=event_json)
    
    def save_model_weights(self,performance):
        weights = self.model.get_weights()
        payload = {'performance': performance, 'weights': weights}
        payloads = json.dumps(payload, cls=NumpyEncoder)
        compress_json.dump(payloads, 'self/model'+str(self.current_round)+'.json.gz')
        
    def save_history(self, performance):
        df = pd.DataFrame(performance)
        loss = df['loss'].iloc[-1]
        self.DB.query_executer("update performance set `"+str(self.id)+"` ='"+str(loss)+"' where round ="+str(self.current_round))
        if(self.current_round == 1):
            df.to_csv('Performance/performance.csv', mode='w',header=True)
        else:
            df.to_csv('Performance/performance.csv', mode='a', header=False)
    
    def check_neighbors(self):
        # neighbors_id = []
        # for i in self.neighbors:
        #     neighbors_id.append(+str(i[0]))
        
        cols = ""
        for i in range(0, len(self.neighbors)-1):
            cols += "`"+str(self.neighbors[i][0])+'`,'
        cols += "`"+str(self.neighbors[-1][0])+"`"
        query = 'select '+cols+' from performance where round ='+str(self.current_round)
        status = self.DB.select_query(query)
        status = status.to_dict()
        return status
    
    def get_model_from_edgeX(self, neighbor_id):
        for neighbor in self.neighbors:
            if(neighbor[0] == int(neighbor_id)):
                try:
                    url = "http://"+neighbor[1]+":59880/api/v2/reading/device/name/Node"

                    response = requests.get(url)
                    data = response.json()
                    payload = data["readings"][0]["value"]
                    compress_json.dump(payload, "neighbors/model"+str(neighbor[0])+".json.gz")
                    # open("neighbors/model"+str(neighbor[0])+".json.gz", "+wb").write(payload)
                    print(neighbor, 'received')
                except:
                    print(neighbor, ' is offline')

    def get_model_from_neighbor(self,neighbor_id):
        for neighbor in self.neighbors:
            if(neighbor[0] == int(neighbor_id)):
                try:
                    url = 'http://'+neighbor[1]+':'+str(neighbor[2])+'/' +'model_to_neighbors?iteration='+str(self.current_round)
                    response = requests.get(url)
                    open("neighbors/model"+str(neighbor[0])+".json.gz", "+wb").write(response.content)
                except:
                    print(neighbor, ' is offline')
    
    def check_status_neighbors(self):
        self.update_log('Waiting for Neighbors',4)
        # status = self.check_neighbors() #status contain all neighbor's status
        received = []
        start_wait_time = time.time()
        while(True):  # while all neighbors not complete the training
            status = self.check_neighbors()
            print(status)
            self.early_stop = self.check_early_stop()
            if(self.early_stop):
                self.update_log("--Early Stopped--",7)
                break
            # status_copy = status.copy()
            for key in status:
                if(key in received):
                    continue
                if(status[key][0] != None):
                    end_wait_time = time.time()
                    wait_time = end_wait_time - start_wait_time
                    try:
                        self.waiting_time[key].append(wait_time)
                    except:
                        self.waiting_time[key] = [wait_time]
                    received.append(key)
                    print('+Receiving from ',key)
                    self.update_log('Receiving Model from '+str(key),5)
                    self.get_model_from_neighbor(key)
                    # self.get_model_from_edgeX(key)
                    # status_copy.pop(key, None)  # remove neighbor after receiving model 
            # status = status_copy.copy()
            if(len(received) == len(self.neighbors)):
                break
            time.sleep(10)
        df_time = pd.DataFrame(self.waiting_time)
        if self.first_iteration:
            df_time.to_csv('Performance/Waiting Time.csv', mode='w', header=True, index=False)
        else:
            df_time.to_csv('Performance/Waiting Time.csv', mode='a', header=False, index=False)
        return True
    
    def average_weights(self, weights):
        layer_mean = np.nanmean(weights, axis=0)
        # layer_mean = np.round(layer_mean, decimals=2)
        if np.isnan(layer_mean).any():
            layer_mean = np.nan_to_num(layer_mean)
        return layer_mean
    
    
    def simple_federated_averaging(self, weights):
        num_clients = len(weights)
        aggregated_weights = [np.zeros_like(weights) for weights in weights[0]]
        
        for i in range(len(aggregated_weights)):
            for client_weights in weights:
                aggregated_weights[i] += client_weights[i]
            aggregated_weights[i] /= num_clients
        return aggregated_weights
    
    def aggregate_weights(self,weights):
        aggregated_weights = []
        for i in range(0, len(weights[0])):    # layer wise aggragation
            t = []
            print('Layer: '+str(i) +' Aggregation')
            for c in range(0, len(weights)): #self.num_of_clients  
                t.append(weights[c][i])                    # extract layer i from neighbor c weights 
            aggregated_weights.append(self.average_weights(np.array(t)))    # aggregate layers
        #self.save_model_json(aggregated_weights)
        return aggregated_weights
    
    
    def load_neighbors_weights(self, selected_neighbors = None):
        print('** Loading neighbors weights **')
        if(selected_neighbors == None):   # This can be used when we apply neighbor selection based on performance 
            selected_neighbors = self.neighbors  # If neighbor selection is not performed select all neighbors 
        all_weights = []
        self.update_log('Aggregating Models', 1)
        for neighbor in selected_neighbors:
            data = compress_json.load("neighbors/model"+str(neighbor[0])+".json.gz")
            data = json.loads(data)
            weights = [neighbor[0], np.array([np.array(i) for i in data['weights']])] # add neighbor id and weights in one list
            # performance =  data['performance']  # save performance for future use (performance of neighbor is not used at this point)
            # df = pd.DataFrame(performance, columns= performance.keys()) # didn't use
            all_weights.append(weights)
        return all_weights 
    
    def check_model_quality(self):
        nan_count = 0
        zero_count = 0
        weight_count  = 0
        
        for weight in self.model.weights:
            weight_array = weight.numpy()
            nan_count += np.isnan(weight_array).sum()
            zero_count += np.count_nonzero(weight_array == 0)
            weight_count += np.size(weight_array)
        avg_nan = np.round(nan_count / weight_count , 2)
        avg_zero = np.round(zero_count / weight_count , 2)
        self.update_log("Model Quality: Zeros: "+ str(avg_zero)+ " NaN: "+str(avg_nan),7)
        return nan_count, zero_count
    
    def aggregate_model(self):
        self.check_status_neighbors()
        print('--------All neighbors completed training--------')
        all_weights = self.load_neighbors_weights()
        all_weights = [weights[1] for weights in all_weights] # extract only weights from returned list 
        all_weights.append(self.model.get_weights())   # append self weights to all weights
        aggregated_weights = self.simple_federated_averaging(all_weights)
        self.model.set_weights(aggregated_weights)
        self.check_model_quality()
        self.test_classification_model_for_precision('after')

    # def get_model_weights_from_edgeX(self):
    #     print('** Loading neighbors weights **')

    #     if(selected_neighbors == None):   # This can be used when we apply neighbor selection based on performance 
    #         selected_neighbors = self.neighbors  # If neighbor selection is not performed select all neighbors 
    #     all_weights = []
    #     self.update_log('Aggregating Models', 1)
    #     for neighbor in selected_neighbors:
    #         data = compress_json.load("neighbors/model"+str(neighbor[0])+".json.gz")
    #         data = json.loads(data)
    #         weights = [neighbor[0], np.array([np.array(i) for i in data['weights']])] # add neighbor id and weights in one list
    #         performance =  data['performance']  # save performance for future use (performance of neighbor is not used at this point)
    #         df = pd.DataFrame(performance, columns= performance.keys()) # didn't use
    #         all_weights.append(weights)
    #     return all_weights 
    
    def aggregate_model_for_RL(self):
        self.check_status_neighbors()
        print('--------All neighbors completed training--------')
        # Get all weights of neighbors
        # all_weights = self.load_neighbors_weights()
        # # aggregate neighbors model one by one and evaluate
        # self_weights = self.model.get_weights()
        # for weights in all_weights:      # all_weights contain neighbor id and weights of that neighbor
        #     neighbor_id = weights[0]
        #     combined_weights = [self_weights, weights[1]]
        #     aggregated_weights = self.aggregate_weights(combined_weights)
        #     self.model.set_weights(aggregated_weights)
            
        # self.model.set_weights(aggregated_weights)
        
        
    def update_log(self,info, msg_id):
        try:
            self.DB.query_executer("insert into sl_log (device_id, info,log_msg_type_id) values("+str(self.id)+",'"+info+"',"+str(msg_id)+")")
        except:
            print('Faild to update Log')
    
    def check_early_stop(self):
        early_stop = self.DB.select_query('SELECT early_stop FROM configuration ORDER BY conf_id DESC LIMIT 1;')
        early_stop = early_stop.values.tolist()
        if(early_stop[0][0] == 1):
            return True
        return False
    
    def start(self):
        # self.update_log('Training Started')
        while(self.current_round <= self.rounds and self.early_stop == False):
                self.early_stop = self.check_early_stop()
                if(self.early_stop):
                    self.update_log("--Early Stopped--",7)
                    break
                self.update_log('Round '+str(self.current_round)+" Started",1)
                self.train_model()
                # check leader node
                # if leader node Wait for Leeader node for aggregated model 
                # else below line 
                self.aggregate_model()
                print('Round '+str(self.current_round)+" Completed")
                self.first_iteration = False
                self.update_log('Round '+str(self.current_round)+" Completed",6)
                self.current_round += 1
        self.update_log("--Training Completed--",0)
    
    def test_model(self,model):
        test_p = model.predict(self.X_test)
        test_mae = mean_absolute_error(self.y_test, test_p)
        return test_mae
    
    def test_classification_model_for_precision(self, test_type):
        y_pred = self.model.predict(self.X_test)
        if y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)
        
        if self.y_test.ndim > 1:
            y_test_labels = self.y_test.argmax(axis=1)
        else:
            y_test_labels = self.y_test

        precisions = precision_score(y_test_labels, y_pred, average=None)
        class_names = ['class'+str(i) for i in range(0, len(precisions))]
        precision_df = pd.DataFrame([precisions], columns=class_names)
        
        if(test_type == 'before'):
            file_name = 'Performance/precision_before.csv'
        else:
            file_name = 'Performance/precision_after.csv'
        
        if self.first_iteration:
            precision_df.to_csv(file_name, mode='w', header=True, index=False)
        else:
            precision_df.to_csv(file_name, mode='a', header=False, index=False)
            
        
    def evaluate_neighbors_models(self):
        self.update_log("Evaluating Neighbors Models",5)
        for neighbor in self.neighbors:
            print(neighbor)
            payload = compress_json.load("neighbors/model"+str(neighbor[0])+".json.gz")
            payload = json.loads(payload)
            weights = np.array([np.array(i) for i in payload['weights']])
            n_model = self.model
            n_model.set_weights(weights)
            loss = None
            try:
                loss = eval("self.ErrorMethods."+self.model_loss)(n_model)
            except:
                print("Not found: ErrorMethods."+self.model_loss)
                self.update_log("Not found: ErrorMethods."+self.model_loss, 0)
            try:
                self.loss_on_neighbor_models[str(neighbor[0])].append(loss)
            except:
                self.loss_on_neighbor_models[str(neighbor[0])] = [loss]
            
        
    def store_waiting_time(self):
        for neighbor_id in self.waiting_time:
            q = 'insert into rl_weights (n_from, n_to, waiting_time, round) values ('+str(neighbor_id)+','+str(self.id)+',"'+str(self.waiting_time[neighbor_id][-1])+'",'+str(self.current_round)+');'
            self.DB.query_executer(q)
            
    def store_loss(self):
        for neighbor_id in self.loss_on_neighbor_models:
            q = 'update rl_weights set loss = "'+str(self.loss_on_neighbor_models[neighbor_id][-1])+'" where n_from = '+str(neighbor_id)+' and n_to= '+str(self.id)+' and round= '+str(self.current_round)+'; '
            self.DB.query_executer(q)
            
    def store_reward(self,reward):
        for neighbor_id in reward:
            q = 'update rl_weights set reward = "'+str(reward[neighbor_id])+'" where n_from = '+str(neighbor_id)+' and n_to= '+str(self.id)+' and round= '+str(self.current_round)+'; '
            self.DB.query_executer(q)
            
    def compute_alpha(self,rewards):
        total = sum(rewards.values())
        weights = {k: np.round(v / total, 2) for k, v in rewards.items()}
        print(weights)
        print('----------------- Alpha Weights ------------------')
        return weights
    
    def compute_reward(self):
        reward = {}
        for neighbor in self.neighbors:
            n_w_time = self.waiting_time[str(neighbor[0])][self.current_round-1]
            n_loss = self.loss_on_neighbor_models[str(neighbor[0])][self.current_round-1]
            reward[neighbor[0]] =np.round(1/(n_w_time + n_loss), 2)
        return reward
    
    def aggregate_weighted(self,weights):
        aggregated_weights = []
        for i in range(0, len(weights[0])):
            t = []
            for c in range(0, len(self.neighbors)): #self.num_of_clients
                t.append(weights[c][i])
            aggregated_weights.append((self.average_weights(t)))
        #self.save_model_json(aggregated_weights)
        return aggregated_weights
    
    def assign_weights_to_neighbor_weights(self,weight, model_weight):
        w_model = []
        if np.isnan(weight):
            weight = 0
        for layer_weights in model_weight:
            layer_weights = np.nan_to_num(layer_weights)
            w_layer = layer_weights * weight
            if np.isnan(w_layer).any():
                print("****************NaN value detected in model layer weights****************")
            w_model.append(w_layer)
        return np.array(w_model)
    
    def assign_weights_to_models(self, alpha, all_neighbor_weights):
        scaled_model_weights = []
        neighbor_weights_dict = {neighbor[0]: neighbor[1] for neighbor in all_neighbor_weights}
        for neighbor_id, weight in alpha.items():
            
            model_weight = neighbor_weights_dict.get(neighbor_id)
            if model_weight is not None:
                scaled_weights = np.array([np.nan_to_num(np.array(layer_weight)) * weight for layer_weight in model_weight])
                scaled_model_weights.append(scaled_weights)
        return scaled_model_weights
    
    def neighbors_weighted_model(self, alpha, all_neighbor_weights):
        weighted_avg_model_weights = None
        neighbor_weights_dict = {neighbor[0]: neighbor[1] for neighbor in all_neighbor_weights}
        for neighbor_id, weight in alpha.items():
            model_weight = neighbor_weights_dict.get(neighbor_id)
            if model_weight is not None:
                scaled_weights = [np.nan_to_num(np.array(layer_weight)) * weight for layer_weight in model_weight]
                if weighted_avg_model_weights is None:
                    weighted_avg_model_weights = np.array([np.zeros_like(layer_weight) for layer_weight in scaled_weights])
                for i, layer_weight in enumerate(scaled_weights):
                    weighted_avg_model_weights[i] += np.array(layer_weight)
        return [np.array(weighted_avg_model_weights)]

    def take_step(self, alpha):
        all_neighbor_weights = self.load_neighbors_weights() # 0 contain neighbor id and 1 contain weights
        scaled_weights = self.neighbors_weighted_model(alpha, all_neighbor_weights) # self.assign_weights_to_models(alpha, all_neighbor_weights)
        # [weights[1] for weights in all_neighbor_weights]
        # weighted_models = []
        # for n_weights in all_neighbor_weights:
        #      n_id = n_weights[0]
        #      n_model_w = n_weights[1]
        #      n_w = alpha[n_id]                    # confused about variable names 
        #      weighted_weights = self.assign_weights_to_neighbor_weights(n_w, n_model_w)
        #      weighted_models.append(weighted_weights)
        
        scaled_weights.append(self.model.get_weights())
        aggregated_weights = self.aggregate_weights(scaled_weights)
        self.model.set_weights(aggregated_weights)
        step_sorted = dict(sorted(alpha.items(), key=lambda x: x[1]))
        step = list(step_sorted.keys())
        # loss = self.test_model(self.model) 
        loss = eval("self.ErrorMethods."+self.model_loss)(self.model)
        step_reward = 1/loss
        return step, step_reward
    
    def update_reward(self, reward, alpha, step_reward):
        updated_reward = {}
        for i in reward:
            updated_reward[i] = reward[i] + alpha[i] * step_reward
        return updated_reward
    
    # reward: reward assigned to each incoming edge of the current node
    # alpha: weights assign to each model of neighbor based on performance and waiting time 
    def learn_with_RL(self):
        while(self.current_round <= self.rounds and self.early_stop == False):
            start_train_time = time.time()
            self.train_model()
            end_train_time = time.time()
            self.check_status_neighbors()   # waiting time computation 
            self.store_waiting_time()
            self.evaluate_neighbors_models()   # Neighbors model evaluation
            self.store_loss()
            reward = self.compute_reward() # initial Reward 
            alpha = self.compute_alpha(reward) # dictionay: neighbor id is key and alpha (as weight) is value
            step, step_reward = self.take_step(alpha)
            self.update_log('Step -> '+ ','.join(str(i) for i in step), 0)
            self.update_log('Reward -> '+ str(step_reward), 0)
            updated_reward = self.update_reward(reward,alpha, step_reward)
            self.store_reward(updated_reward)
            self.update_log('Round '+str(self.current_round)+" Completed",6)
            self.first_iteration = False
            self.current_round += 1
        self.update_log("--Training Completed--",6)
        return ['']
    

    
    def learn_with_PSO(self):
        while(self.current_round <= self.rounds and self.early_stop == False):
            start_train_time = time.time()
            self.train_model()
            end_train_time = time.time()
            self.check_status_neighbors()   # waiting time computation 
            self.store_waiting_time()
            self.evaluate_neighbors_models()   # Neighbors model evaluation
            self.store_loss()
            reward = self.compute_reward() # initial Reward 
            alpha = self.compute_alpha(reward) # dictionay: neighbor id is key and alpha (as weight) is value
            step, step_reward = self.take_step(alpha)
            self.update_log('Step -> '+ ','.join(str(i) for i in step), 0)
            self.update_log('Reward -> '+ str(step_reward), 0)
            updated_reward = self.update_reward(reward,alpha, step_reward)
            self.store_reward(updated_reward)
            self.update_log('Round '+str(self.current_round)+" Completed",6)
            self.first_iteration = False
            self.current_round += 1
        self.update_log("--Training Completed--",6)
        return ['']
    
class ErrorMethods:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def mean_squared_error(self, model):
        test_p = model.predict(self.X_test)
        error = mean_squared_error(self.y_test, test_p)
        return error
    
    def mean_absolute_error(self, model):
        test_p = model.predict(self.X_test)
        error = mean_absolute_error(self.y_test, test_p)
        return error
    
    def sparse_categorical_crossentropy(self, model):
        test_p = model.predict(self.X_test)
        loss = sparse_categorical_crossentropy(self.y_test, test_p)
        error = tf.reduce_mean(loss).numpy()
        return error
    def categorical_crossentropy(self, model):
        test_p = model.predict(self.X_test)
        error = categorical_crossentropy(self.y_test, test_p)
        error = np.mean(error.numpy())
        return error

class FedProxClient:
    def __init__(self, model, loss_fn, lr, mu):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.mu = mu
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    def local_train(self, x, y, global_weights, local_epochs, batch_size):
        """
        FedProx local training
        """
        # Set reference (global) weights
        self.model.set_weights(global_weights)
        w_ref = [tf.convert_to_tensor(w) for w in global_weights]

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(1024).batch(batch_size)

        for _ in range(local_epochs):
            for xb, yb in dataset:
                with tf.GradientTape() as tape:
                    preds = self.model(xb, training=True)
                    loss = self.loss_fn(yb, preds)

                    # Proximal term
                    prox_term = 0.0
                    for w, w0 in zip(self.model.trainable_variables, w_ref):
                        prox_term += tf.reduce_sum(tf.square(w - w0))

                    loss += (self.mu / 2.0) * prox_term

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

        return self.model.get_weights()
















