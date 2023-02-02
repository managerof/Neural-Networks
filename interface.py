from model import *         # import model `nn`
import json

def save_settings(filename):
    # Create the data to be written to the file
    data = {
        "weights1":str(nn.weights1.tolist()),
        "weights2":str(nn.weights2.tolist()),
        "weights3":str(nn.weights3.tolist()),
        "weights4":str(nn.weights4.tolist()),
    }

    # Write the data to the file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
        file.close()

def load_settings(filename):
    # Load the data from the file
    with open(filename, "r") as file:
        data = json.load(file)
        
        data["weights1"] = json.loads(data["weights1"])
        data["weights2"] = json.loads(data["weights2"])
        data["weights3"] = json.loads(data["weights3"])
        data["weights4"] = json.loads(data["weights4"])
        
        nn.weights1 = data["weights1"]
        nn.weights2 = data["weights2"]
        nn.weights3 = data["weights3"]
        nn.weights4 = data["weights4"]
        
        return data


nn = FlowerTester()

model_setting = "model_setting.json"

# Load the settings from the file
settings = load_settings(model_setting)