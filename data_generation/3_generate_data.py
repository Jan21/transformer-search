import pickle
import random
import os


temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")


# Load the snapshots from the specified file
with open(f'{temp_dir}/output_snapshots.pkl', 'rb') as f:
    snapshots = pickle.load(f)


# Set a random seed for reproducibility
random.seed(42)

# Shuffle the snapshots to ensure random distribution
random_snapshots = snapshots.copy()
random.shuffle(random_snapshots)

# Calculate the split point (20% for test, 80% for train)
split_point = int(len(random_snapshots) * 0.2)

# Split the data
test_snapshots = random_snapshots[:split_point]
train_snapshots = random_snapshots[split_point:]

print(f"Total snapshots: {len(snapshots)}")
print(f"Test snapshots: {len(test_snapshots)} ({len(test_snapshots)/len(snapshots)*100:.1f}%)")
print(f"Train snapshots: {len(train_snapshots)} ({len(train_snapshots)/len(snapshots)*100:.1f}%)")

# Display a few examples from each set
print("\nFirst 2 test examples:")

def tokenize_command(command):
    if command[0] == 'N':
        command = 'N { ' + ' , '.join([p.strip()[0] + " : " + p.strip()[3] for p in command[3:-1].split(',')]) + ' }'
    return command
def get_tokens(snapshots,i,j):
    tokens = "Init_state: [ "
    for k,v in snapshots[i][j]['init_state'].items():
        tokens += f"{k} : {v} , "
    tokens = tokens[:-3] + " ] Stack: [ "
    for s in snapshots[i][j]['stack']:
        tokens += " { "
        for k,v in s.items():
            tokens += f"{k} : {v} , "
        tokens = tokens[:-3] + " } , "
    tokens = tokens[:-3] + " ]"
    tokens += " Command: " + tokenize_command(snapshots[i][j]['path'][0])
    return tokens

train_data = []
test_data = []
for i in range(len(train_snapshots)):
    for j in range(len(train_snapshots[i])):
        tokens = get_tokens(train_snapshots,i,j)
        train_data.append(tokens)
for i in range(len(test_snapshots)):
    for j in range(len(test_snapshots[i])):
        tokens = get_tokens(test_snapshots,i,j)
        test_data.append(tokens)



train_data_path = f'{temp_dir}/train_data.pkl'
test_data_path = f'{temp_dir}/test_data.pkl'

# Save the train and test data to pickle files
with open(train_data_path, 'wb') as f:
    pickle.dump(train_data, f)
    
with open(test_data_path, 'wb') as f:
    pickle.dump(test_data, f)

print(f"Train data saved to {train_data_path} ({len(train_data)} samples)")
print(f"Test data saved to {test_data_path} ({len(test_data)} samples)")

# create a new tokenized version for the test set which for each element
# takes only the first snapshot and omits the 'Command' (i.e. the 'path' key).
def get_tokens_without_command(snapshot):
    """Generate a tokenized string for a single snapshot without the command."""
    tokens = "Init_state: [ "
    for k, v in snapshot['init_state'].items():
        tokens += f"{k} : {v} , "
    tokens = tokens[:-3] + " ] Stack: [ "
    for s in snapshot['stack']:
        tokens += " { "
        for k, v in s.items():
            tokens += f"{k} : {v} , "
        tokens = tokens[:-3] + " } , "
    tokens = tokens[:-3] + " ]"
    return tokens

test_first_data = []
# For every element in the test set, take only the first snapshot and convert it to tokens without the command.
for element in test_snapshots:
    if element:  # Ensure the element is not empty
        first_snapshot = element[0]  # Get the first snapshot of this element
        token_str = get_tokens_without_command(first_snapshot)
        test_first_data.append(token_str)

test_first_data_path = f'{temp_dir}/test_first_data.pkl'
with open(test_first_data_path, 'wb') as f:
    pickle.dump(test_first_data, f)

print(f"Test first data saved to {test_first_data_path} ({len(test_first_data)} samples)")