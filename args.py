import json
import os 
import sys
import shutil


def load_params(file = "args.json"):
    """
    This functions outputs a dictionary corresponding to the json file.
    """
    with open(file) as json_file:
        data = json.load(json_file)

    if data["mode"] == "single": 
        data["size_factor"] = 1
    elif  data["mode"] == "pairs":
         data["size_factor"] = 2
    elif  data["mode"] == "quads":
         data["size_factor"] = 4
    else: 
        raise ValueError("choose a correct mode")
    return data

def save_params(params): 
    assert len(sys.argv) > 1 , "Please provide a name to your model"
    answer = None
    if os.path.exists(("save/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".json")):
        answer = input(str(sys.argv[1]) + " already exists. Do you want to overwrite [y/n]: " )
    if answer != "y" and answer != None: 
        sys.exit()
    if answer == "y":
        shutil.rmtree("save/" + str(sys.argv[1]) + "/")
    os.mkdir("save/" + str(sys.argv[1]) + "/")
    with open(("save/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".json"), 'w') as fp:
        json.dump(params,sort_keys=True, indent=4,fp= fp)