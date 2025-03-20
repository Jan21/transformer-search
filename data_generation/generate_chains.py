import networkx as nx
import random
import argparse
import pickle
import matplotlib.pyplot as plt
import ast
from collections import OrderedDict
from tqdm import tqdm
from itertools import product
from collections import defaultdict
import os

parser = argparse.ArgumentParser()
parser.add_argument("--chains_num", type=int, default=5)
parser.add_argument("--nodes_num", type=int, default=7)
parser.add_argument("--max_rules_num", type=int, default=2)
parser.add_argument("--min_rules_num", type=int, default=1)

args = parser.parse_args()

# generate n chains of length m, where n and m are given by the user, default n = 5, m = 10
# add dependencies on the edges such that the edge of one chain depends on the states of later chains (not the previous ones in the fixed order)
# the number of states on which a given edge depends is chosen uniformly from {0,1, 2, 3}..if there are multiple states, they need to be from different chains.

def gen_chains(n=args.chains_num, m=args.nodes_num):
    chains = defaultdict(list)

    for i in range(n):
        G = nx.Graph()
        nodes = [j for j in range(m)]
        edges = list(zip(nodes[:-1], nodes[1:]))

        G.add_edges_from(edges)
        chains[i] = G

    return chains

def add_enabling(chains):
    n = len(chains)
    
    for chain_id in range(n-1):
        G = chains[chain_id]
        
        for edge in G.edges():
            if random.random() < 0.5:
                num_deps = random.randint(args.min_rules_num, args.max_rules_num)
                
                if num_deps > 0:
                    later_chains = list(range(chain_id + 1, n))
                    selected_chains = random.sample(later_chains, min(num_deps, len(later_chains)))
                    
                    enabling_conditions = []
                    for enabling_chain in selected_chains:
                        enabling_G = chains[enabling_chain]
                        enabling_node = random.choice(list(enabling_G.nodes()))
                        enabling_conditions.append({'automata_id':enabling_chain,'node_id':enabling_node})
                    tmp = [enabling_conditions, [{'automata_id':n-1,'node_id':9999}]]
                    
                    G.edges[edge]['enabling'] = tmp
    
    return chains

def get_sample(vector, graphs):
    num_states = len(graphs[0].nodes())
    vec = list(vector)  # [1,2,3,4]
    for i in range(num_states):
        if vec[0] != i:
            yield vec, i

def get_vectors(chains):
    states_per_chain = []
    for chain_id in range(len(chains)):
        states_per_chain.append(list(chains[chain_id].nodes()))
    
    return list(product(*states_per_chain))

def filter_unique_paths(paths_lst):
    if not paths_lst:
        return []
    
    # Keep track of which paths should be excluded
    paths_to_exclude = set()
    
    # For each path, check if it contains any shorter path as a subsequence
    for path in paths_lst:
        for other_path in paths_lst:
            # Skip if same path or if other_path is not shorter
            if path == other_path or len(other_path) >= len(path):
                continue
                
            # Check if other_path is a subsequence of path
            j, k = 0, 0  # j for other_path, k for path
            while j < len(other_path) and k < len(path):
                if other_path[j] == path[k]:
                    j += 1
                k += 1
                
            # If other_path is a subsequence of path, exclude path
            if j == len(other_path):
                paths_to_exclude.add(tuple(path))
                break
    
    # Return all paths except those that contain shorter paths as subsequences
    return [path for path in paths_lst if tuple(path) not in paths_to_exclude]

def solve(curr_states, target_node, graphs, orig_state_len, log=[], depth=0):
    '''
    Solves the problem of reaching target node from current state in graphs by returning the resulting state and log of steps.
    Parameters:
        curr_states: list of current states of automatas; LIST of INTEGERS
        target_node: target node; INTEGER
        graphs: list of graphs; LIST of networkx graphs
        log: list of log messages; LIST of STRINGS
        depth: depth of recursion; INTEGER
    Returns:
        out: states of automatas after solving or None if no solution; LIST of INTEGERS/None
            out: log: list of log messages; LIST of STRINGS
    '''
    indent = '-'*depth*2
    # log.append(f"{'-'*50}")
    curr_states_cpy = curr_states.copy()
    log.append(f"{indent}S {curr_states_cpy} TN {target_node}")
    # get all paths from current state to target node
    paths_lst = list(nx.all_simple_paths(graphs[0], curr_states_cpy[0], target_node))
    # check if a shorter path isn't already in a different longer path
    paths_lst = filter_unique_paths(paths_lst)
    log.append(f"{indent}PL {paths_lst}")
    
    # check if there is a path
    if len(paths_lst) == 0:
        log.append(f"{indent}NOPATH {curr_states} TN {target_node}")
        return None, log
    
    # check if current node equals target node
    if curr_states_cpy[0] == target_node:
        log.append(f"{indent}REND {curr_states_cpy}")
        return curr_states_cpy, log
    
    # path or paths exist and we are not in target node
    for path in paths_lst:
        log.append(f"{indent}P {path}")
        # projdu hrany po ceste (3-4), ziskam enabling
        curr_states_cpy = curr_states.copy()
        rules = graphs[0].get_edge_data(curr_states_cpy[0], path[1])
        # pokud neexistuje enabling, projdu edge
        log.append(f"{indent}RLS {rules}")
        if rules == {}:
            log.append(f"{indent}NOR {curr_states_cpy} TN {target_node}")
            # pokud má path jeden node, tak je to konec
            log.append(f"{indent}{orig_state_len - len(curr_states_cpy)} {'U' if path[1] > curr_states_cpy[0] else 'D'}")
            curr_states_cpy[0] = path[1]
            log.append(f"{indent}NORCH {curr_states_cpy}")

            return solve(curr_states_cpy, target_node, graphs, orig_state_len, log, depth+1)

        # pokud existuje enabling, tak treba automat 1 ma byt ve stavu 2 a automat 2 ma byt ve stavu 3
        # prvni vyresime prvni pravidlo a pak druhe
        else:
            for rule_set in rules['enabling']:
                log.append(f"{indent}RS {rule_set}")

                curr_states_cpy = curr_states.copy()
                found = True

                for rule in sorted(rule_set, key=lambda x: x['automata_id']): # sorted znamena setrizene od nejvice zavisleho po nejmene
                    log.append(f"{indent}RULE {rule}")
                    idx = max(rule['automata_id'] - (orig_state_len - len(curr_states_cpy)), 0)
                    current_sub_state, log = solve(curr_states_cpy[idx:], rule['node_id'], graphs[idx:], orig_state_len, log, depth+1)

                    log.append(f"{indent}RCH {current_sub_state}")
                    if current_sub_state == None:
                        found = False
                        break
                    log.append(f"{indent}SUB {curr_states_cpy}")
                    curr_states_cpy[idx:] = current_sub_state
                    log.append(f"{indent}SUBCH {curr_states_cpy}")

                if found:
                    log.append(f"{indent}SUBF {curr_states_cpy}")
                    log.append(f"{indent}{orig_state_len - len(curr_states_cpy)} {'U' if path[1] > curr_states_cpy[0] else 'D'}")
                    curr_states_cpy[0] = path[1]
                    log.append(f"{indent}SUBSOLVED {curr_states_cpy}")
                    break
                else:
                    continue
            
            if found:
                return solve(curr_states_cpy, target_node, graphs, orig_state_len, log, depth+1)
            else:
                continue
    log.append(f"{indent}UNSOL {curr_states_cpy}")            
    return None, log

def process_graphs(graphs, states, target_nodes, orig_states_len, log=[]):
    log_cpy = log.copy()
    log_cpy.append("*"*30)
    states, log_cpy = solve(states, target_nodes[-1], graphs, orig_states_len, log=log_cpy)

    if len(graphs) == 1:
        return states, target_nodes, log_cpy

    if states is not None:
        log = log_cpy
        found = False
        rnd_target_nodes = [tn for tn in graphs[0].nodes() if tn != states[1]]
        random.shuffle(rnd_target_nodes)
        for i in rnd_target_nodes:
            target_nodes_cpy = target_nodes.copy()
            target_nodes_cpy.append(i)
            state_tmp, target_nodes_cpy, log_tmp = process_graphs(graphs[1:], states[1:], target_nodes_cpy, orig_states_len, log)
            if state_tmp is None:
                continue
            else:
                found = True
                log = log_tmp
                target_nodes = target_nodes_cpy
                states[1:] = state_tmp
                break

        if not found:
            states = None

    return states, target_nodes, log

def parse_log(log):
    dic = {}
    init_state = log[1]
    end_idx = init_state.index("]")
    init_state = init_state[2:end_idx+1]
    # Parse the string into a Python list using ast.literal_eval
    parsed_list = ast.literal_eval(init_state)
    # Create a dictionary with indices as keys and list values as values
    init_state_dict = {i: value for i, value in enumerate(parsed_list)}
    dic['init_state'] = init_state_dict

    # get all indexes of ***
    start_idx = []
    for i in range(len(log)):
        if log[i].startswith("*"*30):
            start_idx.append(i+1)

    target_nodes_lst = []
    for tn in start_idx:
        target_nodes_lst.append(int(log[tn][log[tn].index("TN")+3:]))

    # Parse the string into a Python list using ast.literal_eval
    parsed_list = target_nodes_lst
    # Create a dictionary with indices as keys and list values as values
    parsed_target_states_dict = OrderedDict((i, value) for i, value in enumerate(parsed_list))

    dic['stack'] = parsed_target_states_dict
    
    path = []
    last_num_of_indents = 0
    for item in log:

        curr_num_of_indents = item.count("-")

        if "RS" in item:
            path.append(item)
            
        if " U" in item or " D" in item:
            path.append(item)

        if "RULE" in item:
            path.append("TF")

        if curr_num_of_indents < last_num_of_indents:
            path.append("RL")

        

        last_num_of_indents = curr_num_of_indents

    processed_path = []
    processed_path.append("TF")
    for i, item in enumerate(path):
        if "RS" in item:
            ruleset = ast.literal_eval(item.split("RS")[1])
            result_dict = {item['automata_id']: item['node_id'] for item in ruleset}

            processed_path.append(f"N {result_dict}")
        
        if " U" in item or " D" in item:
            processed_path.append(item.replace("-", ""))

        if "TF" in item:
            processed_path.append("TF")

        if "RL" in item:
            processed_path.append("RL")
            processed_path.append("TF")

    processed_path.append("RL")

    dic['path'] = processed_path

    return dic

chains = add_enabling(gen_chains())

vecs = get_vectors(chains)

id_2_vec = {}
vec_2_id = {}
for i, comb in tqdm(enumerate(vecs)):
    id_2_vec[i] = comb
    vec_2_id[comb] = i

indexes = list(range(len(vecs)))
samples = []

total = 0

error_samples = []
logs = []
for i in tqdm(indexes):
    for sample in get_sample(vecs[i], chains):
        total += 1

        in_states = sample[0]
        in_target_nodes = [sample[1]]

        states, target_nodes, sample_log = process_graphs(list(chains.values()), in_states, in_target_nodes, len(in_states))
            
        if states is not None:
            logs.append({'sample': sample, 'target_nodes': target_nodes, 'log': sample_log})  # Add this ONE sample list to the main list

paths = []

for log in tqdm(logs):
    dict_of_path = parse_log(log['log'])
    paths.append(dict_of_path)

len(paths)

# save paths into pkl file

# Create temp directory if it doesn't exist
temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    print(f"Created directory: {temp_dir}")

with open(f"{temp_dir}/path_data.pkl", "wb") as f:
    pickle.dump(paths, f)

# save graphs into pkl file
with open(f"{temp_dir}/graphs.pkl", "wb") as f:
    pickle.dump(chains, f)