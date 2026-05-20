import yaml, json
from easydict import EasyDict as edict

def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return edict(data)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data