import pickle
import json

print("Start conversion.")

with open("./temp/test_data.pkl", "rb") as f:
  te = pickle.load(f)

with open("./temp/train_data.pkl", "rb") as f:
  tr = pickle.load(f)

with open("./temp/test_first_data.pkl", "rb") as f:
  trf = pickle.load(f)

def convert_to_json_format(input_data):
    """
    Convert the raw data to a list of dictionaries with 'text' keys.

    Args:
        input_data: List of strings containing the raw data

    Returns:
        List of dictionaries with 'text' key containing the original strings
    """
    return [{"text": item} for item in input_data]


# Convert the data to the specified JSON format
test = convert_to_json_format(te)
train = convert_to_json_format(tr)
test_first = convert_to_json_format(trf)

with open('data/test.json', 'w') as f:
    json.dump(test, f, indent=2)

with open('data/train.json', 'w') as f:
    json.dump(train, f, indent=2)

with open('data/test_only_first.json', 'w') as f:  
    json.dump(test_first, f, indent=2)

print("Done conversion.")