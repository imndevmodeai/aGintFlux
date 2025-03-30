#!/usr/bin/env python3
# Copyright (c) 2023-2024 imndevmodeai. All rights reserved.
# See LICENSE file for details.
#
# NOTICE: This file contains proprietary code implementing Sparse Priming Representations (SPR)
# and related knowledge graph processing techniques. The SPR capitalization format (first and last 
# letters capitalized) and associated processing algorithms are original intellectual property.
#
# This code is provided under the terms of the MIT License.
# 
"""
Simple String-Based Knowledge Graph SPR Converter

A minimal script to convert all node names in a knowledge graph to SPR capitalization format.
This script uses simple string replacement without requiring any dependencies.
It is designed to be robust against initial non-KG text and processes line-by-line.
Includes enhanced debugging for node matching.

Usage:
    python string_replace_kg.py unStructured_KG.txt spr_kg.txt

"""

import sys
import re
import os


def format_to_spr_capitalization(text):
    """Convert text to SPR format (first and last letters capitalized)."""
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


def convert_kg_file(input_file, output_file):
    """Convert all node names in the file to SPR capitalization format."""
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Define the node pattern
    node_pattern = re.compile(r"^Node (\d+): (.+)$") # Match from start of line
    
    # Create a mapping of original node labels to SPR capitalized labels
    node_mapping = {}
    node_count = 0
    original_lines = []
    matched_lines_log = []
    unmatched_lines_log = []
    max_log_entries = 5
    last_match_was_node = False

    # Read entire file content first
    print(f"\nReading input file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if not content:
            print("Error: Input file is empty.")
            return False
    except IOError as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Split content into actual lines
    original_lines = content.splitlines() # Use splitlines() to handle different line endings
    print(f"Read {len(original_lines)} lines from the file.")

    # First pass: Process each line to find nodes and build the mapping
    print("\n--- Starting First Pass: Processing Lines for Nodes (Detailed Debug) ---")
    line_debug_limit = 20 # Show detailed debug for first N lines
    
    # Use enumerate on the list of lines now
    for i, line in enumerate(original_lines):
        stripped_line = line.strip()
        
        # --- Detailed Debugging for first few lines ---
        if i < line_debug_limit:
            print(f"\nLine {i+1} Debug:")
            print(f"  Raw line repr: {repr(line)}")
            print(f"  Stripped line: '{stripped_line}'")
            starts_with_node = stripped_line.startswith("Node ") # Check stripped line now
            print(f"  Starts with 'Node ': {starts_with_node}")
        # --- End Detailed Debugging ---
        
        match = node_pattern.match(stripped_line) # Check if the *stripped* line matches node format
        
        # --- More Debugging ---
        if i < line_debug_limit:
            print(f"  Regex match result: {'Match object' if match else 'None'}")
        # --- End More Debugging ---

        if match:
            node_count += 1
            if len(matched_lines_log) < max_log_entries:
                matched_lines_log.append(f"  Line {i+1}: MATCHED: '{stripped_line[:80]}...'") # Log first few matches
            
            node_id = match.group(1)
            node_label = match.group(2).strip()
            spr_label = format_to_spr_capitalization(node_label)
            
            # Store mapping for standalone label and full "Node X: Label" format
            # Make sure keys don't collide if labels are identical
            node_mapping[node_label] = spr_label
            node_mapping[f"Node {node_id}: {node_label}"] = f"Node {node_id}: {spr_label}"
            last_match_was_node = True
        elif last_match_was_node and stripped_line: # Log first few lines *after* a node that don't match
            if len(unmatched_lines_log) < max_log_entries:
                 unmatched_lines_log.append(f"  Line {i+1}: UNMATCHED (after node): '{stripped_line[:80]}...'")
            last_match_was_node = False # Reset flag until next node match
        elif not stripped_line: # Handle blank lines
            last_match_was_node = False # Blank line doesn't count as node
        else: # Line didn't match and wasn't blank after a node
             last_match_was_node = False

    print(f"\n--- End of Detailed Line Debug --- ({line_debug_limit} lines shown)")
    print(f"\nFirst pass complete. Found {node_count} lines matching node pattern.")
    
    print(f"\n--- Node Matching Debug Log (First {max_log_entries}) ---")
    if matched_lines_log:
        print("Matched Lines:")
        for log_entry in matched_lines_log:
            print(log_entry)
    else:
        print("No lines matched the node pattern.")
        
    if unmatched_lines_log:
        print("\nFirst Lines After a Node Match That Did Not Match Pattern:")
        for log_entry in unmatched_lines_log:
            print(log_entry)
    
    print("-----------------------------------------------------\n")

    if not node_mapping:
        print(r"Error: No lines matched the 'Node \d+: ...' format. Cannot perform replacement.")
        return False

    # Sort mapping keys by length (descending) to avoid partial replacements
    sorted_keys = sorted(node_mapping.keys(), key=len, reverse=True)
    
    # Reconstruct content for replacement (use original file content)
    modified_content = content
    
    # Second pass: Replace all occurrences using the sorted keys
    print("--- Starting Second Pass: Replacing Node Names ---")
    replacement_count = 0
    for original in sorted_keys:
        # Ensure we don't try replacing empty strings if they somehow got into mapping
        if not original:
            continue
        
        spr_version_raw = node_mapping[original]
        spr_version_escaped = spr_version_raw.replace('\\', '\\\\') # Escape for re.sub replacement string
        original_escaped_for_regex = re.escape(original) # Escape original string for regex pattern
        
        try:
            # Count occurrences before replacing
            # Use the escaped original string as the pattern
            occurrences = len(re.findall(original_escaped_for_regex, modified_content))
            if occurrences > 0:
                 # Use the escaped original string as the pattern for sub
                 modified_content = re.sub(original_escaped_for_regex, spr_version_escaped, modified_content)
                 replacement_count += occurrences
                 # Optional: Log replacements
                 # if replacement_count < 10: # Log first few replacements
                 #    print(f"  Replaced '{original}' -> '{spr_version_raw}' ({occurrences} times)")
        except re.error as e:
            print(f"Regex error during replacement for key \'{repr(original)}\': {e}")
            # Consider if we should return False here or just skip
            pass # Continue with other replacements
            
    print(f"Second pass complete. Made approximately {replacement_count} replacements.")
    print("-----------------------------------------------------\n")

    # Write output file
    print(f"--- Writing Output File: {output_file} ---")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print("Output file written successfully.")
    except IOError as e:
        print(f"Error writing output file: {e}")
        return False
    print("---------------------------------------------")

    print(f"\nConversion complete! Processed {node_count} potential node definitions.")
    print(f"SPR formatted knowledge graph saved to {output_file}")
    return True


def main():
    """Main function to process command line arguments."""
    # Simple argument handling
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input_file output_file")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\nStarting conversion from {input_file} to {output_file}")
    
    try:
        success = convert_kg_file(input_file, output_file)
        return 0 if success else 1
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        # Print full traceback in debug mode
        if len(sys.argv) > 3 and sys.argv[3] == "--debug":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 