#!/usr/bin/env python3
"""
Knowledge Graph SPR Formatter

This script processes a knowledge graph file and:
1. Converts node names to SPR capitalization format (First and Last letters capitalized)
2. Enhances nodes with metadata required for SPR decompression
3. Exports the formatted knowledge graph

Usage:
    python kg_spr_formatter.py --input unStructured_KG.txt --output spr_formatted_kg.json
"""

import json
import networkx as nx
import argparse
import re
import os

def format_to_spr_capitalization(node_name):
    """Convert a node name to SPR capitalization format (First and Last letters capitalized)."""
    if not node_name or not isinstance(node_name, str):
        return node_name
        
    if len(node_name) <= 1:
        return node_name.upper()
    
    # Handle multi-word concepts with underscores or spaces
    if '_' in node_name or ' ' in node_name:
        separator = '_' if '_' in node_name else ' '
        parts = node_name.split(separator)
        formatted_parts = []
        
        for part in parts:
            if len(part) <= 1:
                formatted_parts.append(part.upper())
            else:
                formatted_parts.append(part[0].upper() + part[1:-1].lower() + part[-1].upper())
        
        return separator.join(formatted_parts)
    
    # Single word formatting
    return node_name[0].upper() + node_name[1:-1].lower() + node_name[-1].upper()

def enhance_node_metadata(node_data, node_name):
    """Add SPR decompression metadata to nodes."""
    node_type = node_data.get('type', 'concept')
    
    # Set SPR rating if missing (scale 1.0-5.0)
    if 'spr' not in node_data:
        # Assign higher SPR values to more complex types
        if node_type in ['brain_teaser', 'puzzle', 'resonant_state']:
            node_data['spr'] = 4.2
        elif node_type in ['quantum_concept', 'advanced_concept']:
            node_data['spr'] = 3.8
        else:
            node_data['spr'] = 3.0
    
    # Add quantum entanglement if missing (0.0-1.0)
    if 'quantum_entanglement' not in node_data:
        if 'quantum' in node_type:
            node_data['quantum_entanglement'] = 0.85
        elif node_type in ['brain_teaser', 'puzzle']:
            node_data['quantum_entanglement'] = 0.7
        else:
            node_data['quantum_entanglement'] = 0.3
    
    # Ensure description exists
    if 'description' not in node_data or not node_data['description']:
        # Generate description based on node name and type
        name_parts = re.sub(r'([A-Z])', r' \1', node_name).strip()
        node_data['description'] = f"A {node_type} related to {name_parts}"
    
    return node_data

def parse_custom_kg_format(file_path):
    """Parse the custom KG format with Node IDs and properly format SPR capitalization."""
    g = nx.DiGraph()
    
    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        if file_size == 0:
            print("ERROR: File is empty!")
            return g
            
        # Open with explicit encoding and error handling
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        print(f"Read {len(content)} characters from file")
        
        # Print a sample of the content to verify format
        content_sample = content[:500].replace('\n', '\\n')
        print(f"Content sample: {content_sample}...")
        
        # Pattern to match node definitions - flexible whitespace using \s+
        node_pattern = r"Node (\d+): ([^\\n]+)\\s+SPR: ([^,]+), \\\"([^\\\"]+)\\\""
        print(f"Using node pattern: {node_pattern}")
        
        # Find node matches
        node_matches = list(re.finditer(node_pattern, content))
        print(f"Initial regex found {len(node_matches)} node matches")
        
        # If no matches, try alternative patterns...
        if len(node_matches) == 0:
            print("Trying alternative patterns...")
            
            # Alternative 1: Flexible whitespace everywhere
            alt_pattern1 = r"Node\\s+(\\d+):\\s+([^\\n]+)\\s+SPR:\\s*([^,]+),\\s*\\\"([^\\\"]+)\\\""
            alt_matches1 = list(re.finditer(alt_pattern1, content))
            print(f"Alternative pattern 1 found {len(alt_matches1)} node matches")
            
            # Alternative 2: Single quotes instead of double quotes
            alt_pattern2 = r"Node (\\d+): ([^\\n]+)\\s+SPR: ([^,]+), '([^']+)'"
            alt_matches2 = list(re.finditer(alt_pattern2, content))
            print(f"Alternative pattern 2 found {len(alt_matches2)} node matches")
            
            # Alternative 3: No quotes at all
            alt_pattern3 = r"Node (\\d+): ([^\\n]+)\\s+SPR: ([^,]+), ([^\\n]+)"
            alt_matches3 = list(re.finditer(alt_pattern3, content))
            print(f"Alternative pattern 3 found {len(alt_matches3)} node matches")
            
            # Alternative 4: Different format altogether (less likely based on sample)
            # alt_pattern4 = r"Node (\\d+): ([^\\n]+).*?SPR: ([^,\\n]+)"
            # alt_matches4 = list(re.finditer(alt_pattern4, content, re.DOTALL))
            # print(f"Alternative pattern 4 found {len(alt_matches4)} node matches")
            
            # Use the best alternative
            if len(alt_matches1) > len(node_matches):
                node_matches = alt_matches1
                node_pattern = alt_pattern1
                print(f"Using alternative pattern 1")
            if len(alt_matches2) > len(node_matches):
                node_matches = alt_matches2
                node_pattern = alt_pattern2
                print(f"Using alternative pattern 2")
            if len(alt_matches3) > len(node_matches):
                node_matches = alt_matches3
                node_pattern = alt_pattern3
                print(f"Using alternative pattern 3")
            # if len(alt_matches4) > len(node_matches):
            #     node_matches = alt_matches4
            #     node_pattern = alt_pattern4
            #     print(f"Using alternative pattern 4")
                
        # Still no matches? Dump some example content
        if len(node_matches) == 0:
            print("\nNo matches found with any pattern. Here are the first 100 lines of the file:")
            lines = content.split('\n')[:100]
            for i, line in enumerate(lines):
                print(f"Line {i+1}: {line}")
                
            # Try a very basic check for any node mentions
            basic_count = content.count("Node ")
            print(f"\nBasic text search found {basic_count} mentions of 'Node '")
            
            # Exit early since we can't parse anything
            return g
        
        # Add all nodes first
        node_ids = {}  # Map node IDs to their SPR-formatted labels
        original_to_spr = {}  # Map original labels to SPR-formatted labels
        node_id_to_original = {} # Map node ID to original label for edge parsing
        
        for i, match in enumerate(node_matches):
            try:
                node_id = int(match.group(1))
                original_label = match.group(2).strip()
                spr_value = float(match.group(3))
                
                # Handle potential missing description - use default if needed
                try:
                    description = match.group(4)
                except IndexError:
                    description = "No description available"
                
                # print(f"Match {i+1}: ID={node_id}, Label={original_label}, SPR={spr_value}") # Reduce verbosity
                
                # Apply SPR capitalization format to the node label
                spr_formatted_label = format_to_spr_capitalization(original_label)
                
                node_ids[node_id] = spr_formatted_label
                original_to_spr[original_label] = spr_formatted_label
                node_id_to_original[node_id] = original_label
                
                # Add node with attributes
                g.add_node(
                    spr_formatted_label,  # Use SPR-formatted label as node identifier
                    id=node_id,
                    original_name=original_label,
                    spr=spr_value,
                    description=description
                )
                
                # Print progress less frequently
                if node_id % 100 == 0 or i < 5:
                    print(f"Processed node {node_id}: {original_label} -> {spr_formatted_label}")
            except Exception as e:
                print(f"Error processing node match {i+1}: {str(e)}")
                print(f"Match content: {match.group(0)[:100]}...")
        
        print(f"\nFinished processing {len(node_ids)} unique nodes.")
        print("Starting edge processing...")
        
        # Now process edges for each node
        edge_count = 0
        
        for node_id, spr_label in node_ids.items():
            try:
                # Find original label from node_id
                original_label = node_id_to_original.get(node_id)
                if not original_label:
                    print(f"Warning: Could not find original label for node ID {node_id}")
                    continue
                    
                # Find edges section for this node using flexible whitespace and end-of-string anchor
                edges_pattern = fr"Node {node_id}: {re.escape(original_label)}\\s+SPR: [^,]+, \\\"[^\\\"]+\\\"\\s+Edges:\\s+(.*?)(?:\\s+Node |\\Z)"
                edges_match = re.search(edges_pattern, content, re.DOTALL)
                
                if edges_match:
                    edges_text = edges_match.group(1).strip()
                    # print(f"Found edges section for node {node_id}: {len(edges_text)} characters") # Reduce verbosity
                    
                    # Debug: Show sample of the edges text for a few nodes
                    # if node_id < 5 and len(edges_text) > 0:
                    #    print(f"Node {node_id} Edges text sample: {edges_text[:100]}...")
                    
                    # Edge pattern with flexible whitespace
                    edge_pattern = r"\\*\\s+Node\\s+(\\d+):\\s+([^,]+),\\s+SPR:\\s+([^,]+),\\s+\\\"([^\\\"]+)\\\""
                    edge_matches = list(re.finditer(edge_pattern, edges_text))
                    
                    # Alternative if first fails (e.g., no SPR/description on edge line)
                    if len(edge_matches) == 0 and len(edges_text) > 0:
                         alt_edge_pattern = r"\\*\\s+Node\\s+(\\d+):\\s+([^,\\n]+)"
                         edge_matches = list(re.finditer(alt_edge_pattern, edges_text))
                         if len(edge_matches) > 0:
                             print(f"Node {node_id}: Used alternative edge pattern, found {len(edge_matches)} edges.")
                    
                    # if node_id < 5: # Reduce verbosity
                    #    print(f"Found {len(edge_matches)} potential edges for node {node_id}")
                    
                    for edge_match in edge_matches:
                        try:
                            target_id = int(edge_match.group(1))
                            
                            # Only add edge if target node exists
                            if target_id in node_ids:
                                target_spr_label = node_ids[target_id]
                                
                                # Extract weight and label if available
                                weight = 1.0  # Default weight
                                label = ""    # Default label
                                
                                try:
                                    # Group indices depend on which pattern matched
                                    if len(edge_match.groups()) >= 4:
                                        weight = float(edge_match.group(3))
                                        label = edge_match.group(4)
                                    # else: only ID and target label matched (alt pattern)
                                except (IndexError, ValueError):
                                    pass  # Use defaults
                                
                                g.add_edge(
                                    spr_label,
                                    target_spr_label,
                                    weight=weight,
                                    label=label
                                )
                                edge_count += 1
                                # if edge_count % 100 == 0 or edge_count < 10: # Reduce verbosity
                                #    print(f"Added edge {edge_count}: {spr_label} -> {target_spr_label}")
                            # else: # Reduce verbosity
                                # print(f"Warning: Target node ID {target_id} not found for edge from node {node_id}")
                        except Exception as e:
                            print(f"Error processing edge: {str(e)}")
                            print(f"Edge match content: {edge_match.group(0)}")
            except Exception as e:
                print(f"Error processing edges for node {node_id}: {str(e)}")
        
        print(f"\nCustom parser finished. Found {len(g.nodes())} nodes and {len(g.edges())} edges.")
        return g
    except Exception as e:
        import traceback
        print(f"Critical error in parser: {str(e)}")
        traceback.print_exc()
        return g

def parse_unstructured_kg(file_path):
    """Parse the unstructured KG file into a NetworkX graph."""
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try the custom parser first for any file
    try:
        print("Attempting to use custom parser...")
        # Check first few lines to see if it matches our expected format
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = ''.join([f.readline() for _ in range(10)])
        
        # Be more lenient in checking for custom format
        if "Node " in first_lines and ("SPR: " in first_lines or "Edges:" in first_lines):
            print("File appears to match custom format, using custom parser")
            return parse_custom_kg_format(file_path)
        else:
            print("File does not appear to match custom format. Falling back to standard parsers.")
    except Exception as e:
        print(f"Custom parser check failed: {e}. Falling back to standard formats.")
    
    # Fall back to original parsing logic for standard formats
    g = nx.DiGraph()
    
    # Determine file format based on extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.json':
            print("Parsing as JSON...")
            # Assume JSON format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                # Standard graph JSON format
                for node_data in data['nodes']:
                    if isinstance(node_data, dict) and 'id' in node_data:
                        node_id = node_data['id']
                        attrs = {k: v for k, v in node_data.items() if k != 'id'}
                        g.add_node(node_id, **attrs)
                    elif isinstance(node_data, str): # Handle list of node IDs
                         g.add_node(node_data)
                
                for edge_data in data['edges']:
                    if isinstance(edge_data, dict) and 'source' in edge_data and 'target' in edge_data:
                        source = edge_data['source']
                        target = edge_data['target']
                        attrs = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
                        # Ensure nodes exist before adding edge
                        if source not in g: g.add_node(source)
                        if target not in g: g.add_node(target)
                        g.add_edge(source, target, **attrs)
            
            elif isinstance(data, dict):
                # Adjacency list format
                for node, neighbors in data.items():
                    if node not in g:
                        g.add_node(node)
                    
                    if isinstance(neighbors, list):
                        for neighbor in neighbors:
                            target = None
                            attrs = {}
                            if isinstance(neighbor, dict) and 'target' in neighbor:
                                target = neighbor['target']
                                attrs = {k: v for k, v in neighbor.items() if k != 'target'}
                            elif isinstance(neighbor, str):
                                target = neighbor
                            
                            if target:
                                if target not in g: g.add_node(target)
                                g.add_edge(node, target, **attrs)
                                
                    elif isinstance(neighbors, dict):
                        for target, attrs in neighbors.items():
                             if target not in g: g.add_node(target)
                             g.add_edge(node, target, **attrs)
    
        elif ext in ['.txt', '.csv', '.edgelist']:
            print(f"Parsing as {ext} edge list...")
            # Try to parse as edge list
            try:
                # Use nx.read_edgelist, handling potential comments and delimiters
                g = nx.read_edgelist(file_path, comments='#', delimiter=',', create_using=nx.DiGraph(), nodetype=str)
                # If successful, great. If not, maybe it's adjacency?
            except Exception as e_edge:
                print(f"Failed to read as standard edgelist ({e_edge}), trying line-by-line.")
                # Fallback to manual line parsing if nx.read_edgelist fails
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try splitting by comma, then space/tab
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 2:
                        parts = [p.strip() for p in line.split()] # Split by whitespace
                        
                    if len(parts) >= 2:
                        source = parts[0]
                        target = parts[1]
                        
                        # Add nodes if they don't exist
                        if source not in g: g.add_node(source)
                        if target not in g: g.add_node(target)
                            
                        # Add edge (ignore extra parts for simplicity here)
                        g.add_edge(source, target)
                    # else: Line doesn't look like an edge, ignore for this simple parser
            
        elif ext == '.gexf':
             print("Parsing as GEXF...")
             g = nx.read_gexf(file_path)
        elif ext == '.graphml':
             print("Parsing as GraphML...")
             g = nx.read_graphml(file_path)
             
        else:
            print(f"Warning: Unsupported file format '{ext}'. Attempting simple edge list parsing.")
            # Try simple edge list as a last resort
            try:
                g = nx.read_edgelist(file_path, comments='#', create_using=nx.DiGraph(), nodetype=str)
            except Exception as e_last_resort:
                 print(f"Error: Could not parse file '{file_path}' with any known format. Error: {e_last_resort}")
                 return nx.DiGraph() # Return empty graph

    except Exception as parse_error:
         print(f"Error parsing file '{file_path}' with standard parsers: {parse_error}")
         return nx.DiGraph() # Return empty graph
            
    print(f"Standard parser found {len(g.nodes())} nodes and {len(g.edges())} edges.")
    return g


def convert_kg_to_spr_format(input_file, output_file):
    """
    Main function to convert knowledge graph to SPR format.
    """
    print(f"Parsing knowledge graph from {input_file}...")
    kg = parse_unstructured_kg(input_file)
    
    # Check if parsing failed
    if kg is None or len(kg.nodes()) == 0:
         print("Error: Knowledge graph parsing failed or resulted in an empty graph. Cannot proceed.")
         return None
         
    print(f"Original graph has {len(kg.nodes())} nodes and {len(kg.edges())} edges")
    
    # Create a new graph with SPR-formatted nodes
    new_kg = nx.DiGraph()
    
    # Track node name mappings
    node_mapping = {}
    
    # Process nodes
    print("Reformatting node names to SPR capitalization and enhancing metadata...")
    processed_nodes = 0
    for node in kg.nodes():
        # Handle potential non-string node IDs from some parsers
        original_node_id = str(node) 
        
        # Format node name to SPR capitalization
        spr_node = format_to_spr_capitalization(original_node_id)
        node_mapping[original_node_id] = spr_node
        
        # Get and enhance node data - handle potential missing data
        node_data = kg.nodes.get(node, {}) # Use .get for safety
        if not isinstance(node_data, dict): # Ensure it's a dict
             node_data = {}
             
        # Add original ID if not present from custom parser
        if 'id' not in node_data and 'original_name' not in node_data :
             try:
                  node_data['id'] = int(original_node_id) # Try converting back if it looks like an int ID
             except ValueError:
                  node_data['original_name'] = original_node_id # Otherwise store original string ID
        elif 'id' in node_data and 'original_name' not in node_data:
             node_data['original_name'] = original_node_id # Store original string ID if numeric ID exists

        enhanced_data = enhance_node_metadata(node_data, spr_node)
        
        # Add node to new graph
        new_kg.add_node(spr_node, **enhanced_data)
        processed_nodes += 1
        if processed_nodes % 200 == 0:
             print(f"  Processed {processed_nodes} nodes...")

    print(f"Finished processing {processed_nodes} nodes.")
    
    # Process edges
    print("Updating edges with new node names...")
    processed_edges = 0
    skipped_edges = 0
    for u, v, data in kg.edges(data=True):
         original_u = str(u)
         original_v = str(v)
         
         if original_u in node_mapping and original_v in node_mapping:
             new_u = node_mapping[original_u]
             new_v = node_mapping[original_v]
             # Ensure edge data is a dict
             edge_data = data if isinstance(data, dict) else {}
             new_kg.add_edge(new_u, new_v, **edge_data)
             processed_edges += 1
             if processed_edges % 500 == 0:
                  print(f"  Processed {processed_edges} edges...")
         else:
              skipped_edges += 1
              # print(f"Warning: Skipping edge ({original_u} -> {original_v}) due to missing node mapping.") # Reduce verbosity

    print(f"Finished processing edges. Added {processed_edges}, skipped {skipped_edges}.")
    
    # Generate statistics
    node_types = {}
    avg_spr = 0
    spr_count = 0
    for node, data in new_kg.nodes(data=True):
        node_type = data.get('type', 'concept')
        node_types[node_type] = node_types.get(node_type, 0) + 1
        if 'spr' in data:
             try:
                  avg_spr += float(data['spr'])
                  spr_count += 1
             except (ValueError, TypeError):
                  pass # Ignore nodes where SPR isn't a valid number
                  
    avg_spr = avg_spr / spr_count if spr_count > 0 else 0

    print("\nKnowledge Graph Statistics:")
    print(f"Total Nodes: {len(new_kg.nodes())}")
    print(f"Total Edges: {len(new_kg.edges())}")
    if spr_count > 0:
         print(f"Average SPR Value: {avg_spr:.3f}")
    print("\nNode Types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  - {node_type}: {count} nodes")
    
    # Export the new graph
    print(f"\nExporting SPR-formatted knowledge graph to {output_file}...")
    
    # Determine output format based on extension
    _, ext = os.path.splitext(output_file)
    ext = ext.lower()
    
    try:
        if ext == '.json':
            # Export as JSON node-link format
            data = nx.node_link_data(new_kg)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        elif ext == '.gexf':
            # Export as GEXF for visualization in tools like Gephi
            nx.write_gexf(new_kg, output_file)
        
        elif ext == '.graphml':
            # Export as GraphML
            nx.write_graphml(new_kg, output_file)
        
        elif ext in ['.csv', '.txt', '.edgelist']:
             # Export as CSV edge list
             nx.write_edgelist(new_kg, output_file, delimiter=',', data=True)
             
        else:
            print(f"Warning: Unrecognized output extension '{ext}'. Defaulting to JSON.")
            output_file = os.path.splitext(output_file)[0] + '.json'
            data = nx.node_link_data(new_kg)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        print(f"Conversion complete! SPR-formatted graph saved to {output_file}")
        return new_kg

    except Exception as export_error:
        print(f"Error exporting graph to {output_file}: {export_error}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert Knowledge Graph to SPR Format")
    parser.add_argument("--input", required=True, help="Path to input knowledge graph file")
    parser.add_argument("--output", required=True, help="Path to output file (.json, .gexf, .graphml, .csv, .txt)")
    args = parser.parse_args()
    
    convert_kg_to_spr_format(args.input, args.output)

if __name__ == "__main__":
    main()