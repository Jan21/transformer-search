from collections import OrderedDict
import ast
import pickle
import os
def generate_snapshots(element):
    """
    Process a single element and generate snapshots for each operation.

    Each snapshot is a dict with:
      - 'init_state': the current state of automats (a dict).
      - 'stack': a list of OrderedDicts representing the current working stack.
      - 'path': a list containing the single operation (the current op).
    """
    snapshots = []

    # Copy the initial state; this will be modified.
    init_state = element['init_state'].copy()
    # The working stack starts with one OrderedDict (copy of the provided stack).
    stack = [element['stack'].copy()]

    def cleanup_stack():
        """Remove any empty OrderedDicts from the beginning of the working stack."""
        while stack and not stack[0]:
            stack.pop(0)

    for op in element['path']:
        # Check if we should skip logging for this operation.
        if op == "TF":
            cleanup_stack()
            if not stack or not stack[0]:
                print("Error: TF operation attempted on an empty stack or empty first OrderedDict.")
                break
            if len(stack[0]) == 1:
                print("Skipping TF operation due to single-element first dict; not logging snapshot.")
                continue  # Skip logging and processing this op entirely

        # Otherwise, record a snapshot for the current op.
        snapshot = {
            'init_state': init_state.copy(),
            'stack': [od.copy() for od in stack],
            'path': [op]
        }
        snapshots.append(snapshot)

        # Process the operation:
        # Movement commands: "X D" or "X U"
        if op.endswith(" D") or op.endswith(" U"):
            parts = op.split()
            automat_index = int(parts[0])
            direction = parts[1]
            if direction == "D":
                if init_state[automat_index] > 0:
                    init_state[automat_index] -= 1
                else:
                    print(f"Error: Automat {automat_index} is already at minimum value 0.")
                    break
            elif direction == "U":
                if init_state[automat_index] < 6:
                    init_state[automat_index] += 1
                else:
                    print(f"Error: Automat {automat_index} is already at maximum value 6.")
                    break

        # TF (Take First) operation:
        # If the first OrderedDict has more than one element, take the first key-value pair.
        elif op == "TF":
            cleanup_stack()
            if not stack or not stack[0]:
                print("Error: TF operation attempted on an empty stack or empty first OrderedDict.")
                break
            first_dict = stack[0]
            key = next(iter(first_dict))
            value = first_dict.pop(key)
            new_dict = OrderedDict([(key, value)])
            # Insert the new OrderedDict at the beginning of the stack.
            stack.insert(0, new_dict)

        # RL operation:
        # Remove the first OrderedDict if its key-value pair(s) match the current init_state.
        elif op == "RL":
            cleanup_stack()
            if not stack:
                print("Error: RL operation attempted on an empty stack.")
                break
            first_dict = stack[0]
            valid = True
            for key, value in first_dict.items():
                if init_state.get(key) != value:
                    valid = False
                    break
            if valid:
                stack.pop(0)
            else:
                print("Error: RL operation mismatch between the stack and init_state.")
                break

        # N {…} operation:
        # Create a new OrderedDict from the given key-value pair(s) and insert it at the beginning.
        elif op.startswith("N "):
            dict_str = op[2:].strip()
            try:
                new_pair = ast.literal_eval(dict_str)
                if not isinstance(new_pair, dict) or not (1 <= len(new_pair) <= 2):
                    print("Error: N operation dictionary must contain one or two key-value pairs.")
                    break
                cleanup_stack()
                # Insert a new OrderedDict containing the new pair(s) at the beginning.
                stack.insert(0, OrderedDict(new_pair.items()))
            except Exception as e:
                print(f"Error parsing dictionary in N operation: {e}")
                break

        else:
            print(f"Error: Unknown operation '{op}'.")
            break

    return snapshots


def main():
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")

    # Specify the input and output pickle filenames directly in the code.
    input_pickle_file = f"{temp_dir}/path_data.pkl"  # Change this to your input file name.
    output_pickle_file = f"{temp_dir}/output_snapshots.pkl"  # Change this if you prefer another output file name.

    try:
        with open(input_pickle_file, "rb") as f:
            elements = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    if not isinstance(elements, list):
        print("Error: Input pickle file does not contain a list of elements.")
        return

    all_snapshots = []
    for i, element in enumerate(elements):
        snapshots = generate_snapshots(element)
        if snapshots is not None:
            all_snapshots.append(snapshots)
        else:
            print(f"Element {i} skipped due TF operation with single-element first dict.")

    try:
        with open(output_pickle_file, "wb") as f:
            pickle.dump(all_snapshots, f)
        print(f"Successfully generated snapshots for {len(elements)} elements and saved to {output_pickle_file}")
    except Exception as e:
        print(f"Error writing output pickle file: {e}")


if __name__ == "__main__":
    main()
