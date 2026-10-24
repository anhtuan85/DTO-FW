# DTO-FW
Dynamic Topology Optimization based on Frank-Wolfe Algorithm for Decentralized Online Federated Learning

## Installation

* Clone this repository
* Install requirements:
```
pip install -r requirements.txt
```

## Datasets

Check and run the source code in folder `Dataset-generation`

## Run experiment

1. Run the non-IID dataset generation in folder `Dataset-generation` and move each data part to each node

2. Move the DFLNode source code to edge node

3. Run folder DFLManagment in your PC
```
python ./manage.py runserver 0.0.0.0:8000
```

4. Run folder DFLNode in each edge node
```
python App.py
```