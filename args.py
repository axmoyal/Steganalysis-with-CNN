import json
from os import path
import sys


def load_params():
	"""
	This functions outputs a dictionary corresponding to the json file.
	"""

	with open("args.json") as json_file:
		data = json.load(json_file)

	return data

def save_params(params): 
    assert len(sys.argv) > 1 , "Please provide a name to your model"
    answer = None
    if path.exists(("save/" + str(sys.argv[1]) + ".json")):
        answer = input(str(sys.argv[1]) + " already exists. Do you want to overwrite [y/n]: " )
    if answer != "y" and answer != None: 
        sys.exit() 
    with open("save/" + str(sys.argv[1]) + ".json", 'w') as fp:
        json.dump(params,sort_keys=True, indent=4,fp= fp)