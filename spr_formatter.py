import re
import argparse

def distill_to_spr_name(core_concept: str) -> str:
    """
    Distills a core concept string into an SPR-compliant name.

    Capitalizes the first and last words/parts (if multiple) and joins them,
    lowercasing intermediate words/parts. Handles potential underscores.

    Args:
        core_concept: The string representing the core concept.

    Returns:
        An SPR-formatted string. Returns "UnnamedSPR" if the input is empty or None.
    """
    if not core_concept:
        return "UnnamedSPR"

    # Treat underscores as spaces for word separation
    words = core_concept.replace("_", " ").split()

    if not words:
        return "UnnamedSPR" # Handle cases like "____"

    if len(words) == 1:
        # Single word: Capitalize first and last letter
        word = words[0]
        if len(word) <= 1:
             return word.upper() # Handle single letter or empty string after split
        return word[0].upper() + word[1:-1].lower() + word[-1].upper() if len(word) > 1 else word.upper()
    else:
        # Multiple words: Capitalize first and last word, lowercase others
        spr_parts = [words[0].capitalize()] # Capitalize first word
        spr_parts.extend(word.lower() for word in words[1:-1]) # Lowercase middle words
        spr_parts.append(words[-1].capitalize()) # Capitalize last word
        return "".join(spr_parts)


def format_nodes_to_spr(knowledge_graph_text: str) -> str:
    """
    Formats Node names in the knowledge graph text to Sparse Priming Representations (SPRs).

    Identifies "Node X: Concept Name" lines followed by "SPR: ..., "Concept Name""
    and replaces the concept name in the "Node" line with its SPR format.

    Args:
        knowledge_graph_text: The text output of the knowledge graph.

    Returns:
        The modified knowledge graph text with SPR-formatted Node names.
    """
    lines = knowledge_graph_text.splitlines()
    modified_lines = []
    node_info_buffer = {} # Store info temporarily {node_number: original_node_line_index}

    for i, line in enumerate(lines):
        node_match = re.match(r"^(Node\s+(\d+)):(.*)", line) # Match "Node X:" and capture concept name part
        spr_line_match = re.match(r"^\s*SPR:\s*([\d\.]+),\s*\"(.*?)\"", line) # Match "SPR: value, "Concept Name""

        if node_match:
            node_prefix = node_match.group(1) # "Node X"
            node_number = node_match.group(2) # "X"
            # Store the index of this Node line, anticipating the SPR line might be next
            node_info_buffer[node_number] = {'index': i, 'prefix': node_prefix}
            modified_lines.append(line) # Add original line for now
            # print(f"DEBUG: Found Node {node_number} at index {i}. Buffer: {node_info_buffer}")
        elif spr_line_match and node_info_buffer:
            # Found an SPR line, check if it corresponds to the most recent Node in buffer
            # We assume the SPR line immediately follows or is very close to its Node line
            # A more robust approach might involve looking back further or stricter pairing logic
            # For now, let's assume it relates to the *last* node number added to buffer
            # (This might need adjustment if nodes and SPRs aren't strictly sequential)
            last_node_num = max(node_info_buffer.keys(), key=int) # Simplistic assumption
            if last_node_num in node_info_buffer:
                concept_name = spr_line_match.group(2)
                spr_formatted_name = distill_to_spr_name(concept_name)
                node_line_index = node_info_buffer[last_node_num]['index']
                node_prefix = node_info_buffer[last_node_num]['prefix']

                # Modify the previously added Node line
                modified_lines[node_line_index] = f"{node_prefix}: {spr_formatted_name}"
                # print(f"DEBUG: Modifying line {node_line_index} for Node {last_node_num} to: {modified_lines[node_line_index]}")

                # Add the current SPR line (it might also contain the concept name,
                # but requirements are just to change the Node line)
                modified_lines.append(line)

                # Clear the buffer for this node as it's processed
                del node_info_buffer[last_node_num]
            else:
                 modified_lines.append(line) # SPR line doesn't match buffered node? Append as is.
        else:
            modified_lines.append(line) # Keep other lines (Edges, blank lines, etc.)

    return "\\n".join(modified_lines)

def main():
    """
    Main function to handle command-line arguments for file processing.
    """
    parser = argparse.ArgumentParser(description="Format Node names in a knowledge graph file to SPR format.")
    parser.add_argument("input_file", help="Path to the input knowledge graph text file.")
    parser.add_argument("-o", "--output_file", help="Path to the output file. If not provided, prints to stdout.")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            knowledge_graph_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    modified_text = format_nodes_to_spr(knowledge_graph_text)

    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(modified_text)
            print(f"Formatted knowledge graph written to: {args.output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print(modified_text)

if __name__ == "__main__":
    main() 