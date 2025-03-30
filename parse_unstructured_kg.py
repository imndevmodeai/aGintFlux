import re
import json

def parse_kg_text(input_filepath, output_filepath):
    """
    Parses an unstructured text file containing KG definitions mixed with conversation,
    extracts nodes, SPRs, and edges, and outputs a structured JSON file.
    """
    nodes = {}
    current_node_id = None

    # Regex patterns
    node_pattern = re.compile(r'^Node (\d+): (.*)')
    spr_pattern = re.compile(r'^SPR: ([0-9.]+), \"(.*)\"') # Match literal \"
    edge_pattern = re.compile(r'^\* Node (\d+): (.*?), SPR: ([0-9.]+), \"(.*)\"') # Match literal \"
    # Simple check for lines likely being conversational filler or instructions
    filler_pattern = re.compile(r'^(?:<\|eot_id\|>|<\|start_header_id\|>|assistant<\|end_header_id\|>|Please type|Note:|To further enhance|Next, we can integrate|Finally, we can integrate|With these additional components|To further improve|Explore<\|eot_id\|>|Integrate<\|eot_id\|>|\.\s*$|\s*$|\"explore\"|\"integrate\"|\"stop\"|\"continue\")', re.IGNORECASE) # Match literal \"


    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()

                # Skip empty lines and obvious filler/instructions
                if not line or filler_pattern.match(line):
                    # Reset current_node_id if we encounter filler after processing edges
                    if current_node_id and (line.startswith('<|eot_id|>') or line.startswith('assistant<|end_header_id|>')):
                         # Helps reset context between conversational turns in the log
                         current_node_id = None
                    continue

                # Match Node Definition
                node_match = node_pattern.match(line)
                if node_match:
                    node_id = int(node_match.group(1))
                    node_name = node_match.group(2).strip()
                    # Avoid duplicates - only add if not already seen
                    if node_id not in nodes:
                        nodes[node_id] = {
                            "node_id": node_id,
                            "node_name": node_name,
                            "spr_number": None,
                            "spr_label": None,
                            "edges": []
                        }
                        current_node_id = node_id
                    else:
                        # If node exists, update current_node_id but don't overwrite unless needed
                        # This handles cases where a node is mentioned again before adding edges
                        current_node_id = node_id
                    continue # Move to next line after processing node

                # Match SPR Definition (associated with the current node)
                spr_match = spr_pattern.match(line)
                if spr_match and current_node_id is not None and current_node_id in nodes and nodes[current_node_id]["spr_number"] is None:
                     # Only add SPR if it hasn't been set for this node yet
                    spr_number_str = spr_match.group(1)
                    spr_label = spr_match.group(2)
                    try:
                        spr_number = float(spr_number_str)
                        nodes[current_node_id]["spr_number"] = spr_number
                        nodes[current_node_id]["spr_label"] = spr_label
                    except ValueError:
                        print(f"Warning: Could not parse SPR number '{spr_number_str}' for node {current_node_id}. Skipping SPR.")
                    continue # Move to next line

                # Match Edge Definition (associated with the current node)
                edge_match = edge_pattern.match(line)
                if edge_match and current_node_id is not None and current_node_id in nodes:
                    target_id = int(edge_match.group(1))
                    target_name = edge_match.group(2).strip()
                    target_spr_number_str = edge_match.group(3)
                    target_spr_label = edge_match.group(4)

                    try:
                        target_spr_number = float(target_spr_number_str)
                        edge_data = {
                            "target_id": target_id,
                            "target_name": target_name,
                            "target_spr_number": target_spr_number,
                            "target_spr_label": target_spr_label
                        }
                        # Avoid adding duplicate edges for the same node
                        if edge_data not in nodes[current_node_id]["edges"]:
                            nodes[current_node_id]["edges"].append(edge_data)
                    except ValueError:
                         print(f"Warning: Could not parse target SPR number '{target_spr_number_str}' for edge from node {current_node_id} to {target_id}. Skipping edge.")
                    continue # Move to next line

                # If line doesn't match any pattern and isn't filler, it might be unexpected content
                # Consider adding logging here if debugging further
                # print(f"Info: Skipping unrecognized line: {line}")


    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return

    # Convert the dictionary of nodes to a list for JSON output
    output_data = list(nodes.values())

    # Sort by node_id for consistency
    output_data.sort(key=lambda x: x['node_id'])

    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(output_data, outfile, indent=4)
        print(f"Successfully parsed and wrote structured KG to {output_filepath}")
    except IOError as e:
        print(f"Error writing to output file {output_filepath}: {e}")
    except Exception as e:
        print(f"An error occurred during JSON writing: {e}")


if __name__ == "__main__":
    input_file = "blueprints/unStructured_KG.txt"
    output_file = "blueprints/structured_kg.json"
    parse_kg_text(input_file, output_file) 