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