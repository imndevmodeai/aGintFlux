#!/usr/bin/env python3
"""
SPR Capitalization Converter

This script converts any knowledge graph to use SPR capitalization format 
(first and last letters capitalized) for all node names.

Supports multiple input/output formats including:
- JSON (nodes/edges format)
- GraphML
- GEXF
- NetworkX pickle
- CSV edge list

Usage:
    python spr_converter.py --input kg.json --output spr_kg.json --format json

"""

import os
import sys
import json
import argparse
import networkx as nx
from typing import Dict, Any, List, Tuple


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


def convert_json_kg(input_file: str, output_file: str) -> Tuple[int, int]:
    """
    Convert a JSON-format knowledge graph to use SPR capitalization.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the converted JSON file
        
    Returns:
        Tuple of (number of nodes converted, number of edges converted)
    """
    # Load the knowledge graph
    with open(input_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    # Check if this is a nodes/edges format or adjacency list format
    if isinstance(kg_data, dict) and 'nodes' in kg_data and 'edges' in kg_data:
        # Process nodes/edges format
        
        # Map from original node IDs to SPR node IDs
        node_mapping = {}
        
        # Convert nodes
        for node in kg_data['nodes']:
            # Handle different node ID formats
            if 'id' in node:
                orig_id = node['id']
            elif 'name' in node:
                orig_id = node['name']
            else:
                # Use first field as ID
                orig_id = next(iter(node.values()))
            
            # Generate SPR capitalized ID
            spr_id = format_to_spr_capitalization(str(orig_id))
            node_mapping[orig_id] = spr_id
            
            # Update the node ID in the data
            if 'id' in node:
                node['id'] = spr_id
                
                # Store original ID for reference
                node['original_id'] = orig_id
            elif 'name' in node:
                node['name'] = spr_id
                node['original_name'] = orig_id
            
            # Also capitalize the label if present
            if 'label' in node:
                node['original_label'] = node['label']
                node['label'] = format_to_spr_capitalization(node['label'])
        
        # Convert edges
        for edge in kg_data['edges']:
            if 'source' in edge and 'target' in edge:
                # Store original values
                edge['original_source'] = edge['source']
                edge['original_target'] = edge['target']
                
                # Update to SPR format
                if edge['source'] in node_mapping:
                    edge['source'] = node_mapping[edge['source']]
                if edge['target'] in node_mapping:
                    edge['target'] = node_mapping[edge['target']]
    
    elif isinstance(kg_data, dict):
        # Process adjacency list format
        new_kg_data = {}
        node_mapping = {}
        
        # Create mapping from original node IDs to SPR node IDs
        for node in kg_data:
            spr_node = format_to_spr_capitalization(str(node))
            node_mapping[node] = spr_node
        
        # Create new adjacency list with SPR capitalized IDs
        for node, edges in kg_data.items():
            spr_node = node_mapping[node]
            
            if isinstance(edges, list):
                # List of targets
                new_edges = []
                for edge in edges:
                    if isinstance(edge, dict):
                        # Edge with attributes
                        new_edge = edge.copy()
                        if 'target' in new_edge:
                            new_edge['original_target'] = new_edge['target']
                            if new_edge['target'] in node_mapping:
                                new_edge['target'] = node_mapping[new_edge['target']]
                        new_edges.append(new_edge)
                    else:
                        # Simple edge (just target)
                        if edge in node_mapping:
                            new_edges.append(node_mapping[edge])
                        else:
                            new_edges.append(edge)
                            
                new_kg_data[spr_node] = new_edges
            else:
                # Dictionary of targets to attributes
                new_edges = {}
                for target, attrs in edges.items():
                    if target in node_mapping:
                        new_edges[node_mapping[target]] = attrs
                    else:
                        new_edges[target] = attrs
                        
                new_kg_data[spr_node] = new_edges
        
        kg_data = new_kg_data
    
    # Save the converted graph
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2)
    
    # Count nodes and edges for reporting
    if isinstance(kg_data, dict) and 'nodes' in kg_data and 'edges' in kg_data:
        return len(kg_data['nodes']), len(kg_data['edges'])
    elif isinstance(kg_data, dict):
        # Count edges in adjacency list
        edge_count = 0
        for node, edges in kg_data.items():
            if isinstance(edges, list):
                edge_count += len(edges)
            elif isinstance(edges, dict):
                edge_count += len(edges)
        return len(kg_data), edge_count
    
    return 0, 0


def convert_networkx_kg(input_file: str, output_file: str, format_type: str) -> Tuple[int, int]:
    """
    Convert a knowledge graph using NetworkX to use SPR capitalization.
    
    Args:
        input_file: Path to the input file
        output_file: Path to save the converted file
        format_type: The format type (graphml, gexf, pkl, etc.)
        
    Returns:
        Tuple of (number of nodes converted, number of edges converted)
    """
    # Load the graph based on format
    if format_type == 'graphml':
        G = nx.read_graphml(input_file)
    elif format_type == 'gexf':
        G = nx.read_gexf(input_file)
    elif format_type == 'pkl':
        import pickle
        with open(input_file, 'rb') as f:
            G = pickle.load(f)
    elif format_type == 'csv':
        G = nx.read_edgelist(input_file, delimiter=',', create_using=nx.DiGraph())
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    # Create a new graph with SPR capitalized node IDs
    G_spr = nx.DiGraph()
    
    # Map original node IDs to SPR capitalized IDs
    node_mapping = {}
    
    # Process nodes
    for node, attrs in G.nodes(data=True):
        spr_node = format_to_spr_capitalization(str(node))
        node_mapping[node] = spr_node
        
        # Copy node attributes and add original ID
        new_attrs = attrs.copy()
        new_attrs['original_id'] = node
        
        # Also capitalize label if present
        if 'label' in new_attrs:
            new_attrs['original_label'] = new_attrs['label']
            new_attrs['label'] = format_to_spr_capitalization(new_attrs['label'])
        
        G_spr.add_node(spr_node, **new_attrs)
    
    # Process edges
    for u, v, attrs in G.edges(data=True):
        spr_u = node_mapping[u]
        spr_v = node_mapping[v]
        
        # Copy edge attributes
        new_attrs = attrs.copy()
        new_attrs['original_source'] = u
        new_attrs['original_target'] = v
        
        # Capitalize edge label if present
        if 'label' in new_attrs:
            new_attrs['original_label'] = new_attrs['label']
            new_attrs['label'] = format_to_spr_capitalization(new_attrs['label'])
        
        G_spr.add_edge(spr_u, spr_v, **new_attrs)
    
    # Save the converted graph based on format
    if format_type == 'graphml':
        nx.write_graphml(G_spr, output_file)
    elif format_type == 'gexf':
        nx.write_gexf(G_spr, output_file)
    elif format_type == 'pkl':
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(G_spr, f)
    elif format_type == 'csv':
        nx.write_edgelist(G_spr, output_file, delimiter=',')
    
    return len(G_spr.nodes()), len(G_spr.edges())


def main():
    """Main function to parse arguments and convert the knowledge graph."""
    parser = argparse.ArgumentParser(description="Convert knowledge graph to SPR capitalization format")
    parser.add_argument("--input", required=True, help="Path to input knowledge graph file")
    parser.add_argument("--output", required=True, help="Path to save the converted knowledge graph")
    parser.add_argument("--format", default="auto", choices=["auto", "json", "graphml", "gexf", "pkl", "csv"],
                        help="Format of the input/output files (default: auto-detect from extension)")
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    
    # Auto-detect format if not specified
    format_type = args.format
    if format_type == "auto":
        _, ext = os.path.splitext(args.input)
        ext = ext.lower().lstrip('.')
        
        if ext == 'json':
            format_type = 'json'
        elif ext == 'graphml':
            format_type = 'graphml'
        elif ext == 'gexf':
            format_type = 'gexf'
        elif ext == 'pkl':
            format_type = 'pkl'
        elif ext in ['csv', 'txt']:
            format_type = 'csv'
        else:
            print(f"Warning: Could not auto-detect format from extension '{ext}'. Defaulting to 'json'.")
            format_type = 'json'
    
    print(f"Converting knowledge graph from {args.input} to {args.output} (format: {format_type})")
    
    # Convert based on format
    try:
        if format_type == 'json':
            node_count, edge_count = convert_json_kg(args.input, args.output)
        else:
            node_count, edge_count = convert_networkx_kg(args.input, args.output, format_type)
        
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