import json
def load_params():
	"""
	This functions outputs a dictionary corresponding to the json file.
	"""

	with open("args.json") as json_file:
		data = json.load(json_file)

	# for key in data.keys():
	# 	data[key] = data[key]["data"]
	# 	if type(data[key]) == list:
	# 		data[key] = np.array(data[key])

	return data