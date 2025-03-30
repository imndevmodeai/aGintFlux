#!/usr/bin/env python3
"""
Custom KG Format to SPR Capitalization Converter

This script is designed specifically to convert knowledge graphs in the format:

Node 1: System Overview

SPR: 0.001, "System Overview"

Edges:

* Node 2: System Architecture, SPR: 0.010, "Architecture"
* Node 3: System Requirements, SPR: 0.015, "Requirements"

It extracts all nodes, applies SPR capitalization (first and last letters capitalized),
and outputs a new file with the converted format.

Usage:
    python converter_for_custom_format.py --input unStructured_KG.txt --output spr_kg.txt

"""

import os
import re
import sys
import argparse
from typing import Dict, Any, List, Tuple, Set


def format_to_spr_capitalization(text: str) -> str:
    """
    Convert text to SPR capitalization format (first and last letters capitalized).
    
    Args:
        text: The original text to convert
        
    Returns:
        The text with first and last letters capitalized
    """
    if not text or len(text) <= 1:
        return text.upper() if text else ""
    
    # Handle multi-word text
    words = text.split()
    spr_words = []
    
    for word in words:
        if len(word) <= 1:
            spr_words.append(word.upper())
        else:
            spr_words.append(word[0].upper() + word[1:-1].lower() + word[-1].upper())
    
    return " ".join(spr_words)


def convert_custom_kg(input_file: str, output_file: str) -> Tuple[int, int]:
    """
    Convert a custom format knowledge graph to use SPR capitalization.
    
    Args:
        input_file: Path to the input file
        output_file: Path to save the converted file
        
    Returns:
        Tuple of (number of nodes converted, number of edges converted)
    """
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Create a mapping of original node labels to SPR capitalized labels
    node_mapping = {}
    
    # Extract all nodes
    node_pattern = re.compile(r"Node (\d+): ([^\n]+)")
    node_matches = node_pattern.finditer(content)
    
    # First pass: build node mapping
    for match in node_matches:
        node_id = match.group(1)
        node_label = match.group(2).strip()
        spr_label = format_to_spr_capitalization(node_label)
        node_mapping[node_label] = spr_label
        node_mapping[f"Node {node_id}: {node_label}"] = f"Node {node_id}: {spr_label}"
    
    print(f"Found {len(node_mapping) // 2} unique nodes")  # Divide by 2 because we store two keys per node
    
    # Second pass: replace all occurrences in the content
    modified_content = content
    
    # Sort keys by length in descending order to avoid partial replacements
    sorted_keys = sorted(node_mapping.keys(), key=len, reverse=True)
    
    # Replace all occurrences
    for original in sorted_keys:
        spr_version = node_mapping[original]
        modified_content = modified_content.replace(original, spr_version)
    
    # Count edge replacements
    edge_pattern = re.compile(r"\* Node (\d+): ([^,]+)")
    edge_matches = edge_pattern.finditer(content)
    edge_count = sum(1 for _ in edge_matches)
    
    # Write the modified content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    return len(node_mapping) // 2, edge_count  # Divide by 2 because we store two keys per node


def convert_custom_kg_to_json(input_file: str, output_file: str) -> Tuple[int, int]:
    """
    Convert a custom format knowledge graph to a JSON file with SPR capitalization.
    
    Args:
        input_file: Path to the input file
        output_file: Path to save the converted JSON file
        
    Returns:
        Tuple of (number of nodes converted, number of edges converted)
    """
    import json
    import networkx as nx
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Extract all nodes with SPR and descriptions
    node_pattern = re.compile(r"Node (\d+): ([^\n]+)\n\nSPR: ([^,]+), \"([^\"]+)\"")
    node_matches = node_pattern.finditer(content)
    
    # Process nodes
    for match in node_matches:
        node_id = int(match.group(1))
        original_label = match.group(2).strip()
        spr_value = float(match.group(3))
        description = match.group(4)
        
        # Apply SPR capitalization
        spr_label = format_to_spr_capitalization(original_label)
        
        # Add node to graph with attributes
        G.add_node(
            spr_label,
            id=node_id,
            original_name=original_label,
            spr=spr_value,
            description=description
        )
    
    # Process edges
    edge_count = 0
    
    for node_id in range(1, len(G.nodes()) + 1):
        # Find the original label for this node ID
        original_label = next((data['original_name'] for n, data in G.nodes(data=True) 
                              if data.get('id') == node_id), None)
        
        if not original_label:
            continue
        
        # Find the SPR label for this node
        spr_label = next((n for n, data in G.nodes(data=True) 
                         if data.get('original_name') == original_label), None)
        
        if not spr_label:
            continue
        
        # Find edges section for this node
        edges_pattern = rf"Node {node_id}: {re.escape(original_label)}\n\nSPR: [^,]+, \"[^\"]+\"\n\nEdges:\n\n(.*?)(?:\n\nNode |\Z)"
        edges_match = re.search(edges_pattern, content, re.DOTALL)
        
        if edges_match:
            edges_text = edges_match.group(1)
            edge_pattern = re.compile(r"\* Node (\d+): ([^,]+)")
            edge_matches = edge_pattern.finditer(edges_text)
            
            for edge_match in edge_matches:
                target_id = int(edge_match.group(1))
                
                # Find the original label for the target node
                target_original_label = next((data['original_name'] for n, data in G.nodes(data=True) 
                                            if data.get('id') == target_id), None)
                
                if not target_original_label:
                    continue
                
                # Find the SPR label for the target node
                target_spr_label = next((n for n, data in G.nodes(data=True) 
                                       if data.get('original_name') == target_original_label), None)
                
                if not target_spr_label:
                    continue
                
                # Add edge to graph
                G.add_edge(
                    spr_label,
                    target_spr_label,
                    weight=1.0,
                    source_id=node_id,
                    target_id=target_id
                )
                edge_count += 1
    
    # Export as JSON
    data = {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes()],
        "edges": [{"source": u, "target": v, **G[u][v]} for u, v in G.edges()]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported to JSON format with {len(G.nodes())} nodes and {edge_count} edges")
    return len(G.nodes()), edge_count


def main():
    """Main function to parse arguments and convert the knowledge graph."""
    parser = argparse.ArgumentParser(description="Convert custom format knowledge graph to SPR capitalization")
    parser.add_argument("--input", required=True, help="Path to input knowledge graph file")
    parser.add_argument("--output", required=True, help="Path to save the converted knowledge graph")
    parser.add_argument("--to-json", action="store_true", help="Convert to JSON format instead of the original format")
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    
    print(f"Converting knowledge graph from {args.input} to {args.output}")
    
    try:
        if args.to_json:
            node_count, edge_count = convert_custom_kg_to_json(args.input, args.output)
        else:
            node_count, edge_count = convert_custom_kg(args.input, args.output)
        
        print(f"Conversion complete! Processed {node_count} nodes and {edge_count} edges.")
        print(f"SPR formatted knowledge graph saved to {args.output}")
        return 0
    except Exception as e:
        import traceback
        print(f"Error converting knowledge graph: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 