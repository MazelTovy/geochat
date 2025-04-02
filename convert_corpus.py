#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert NYC Schools Data to FlashRAG Corpus Format
This script converts NYC Schools dataset JSON files to the corpus format required by FlashRAG
"""

import os
import json
import glob
import argparse
import sys


def convert_school_data(corpus_dir, output_file):
    """
    Convert NYC Schools data to FlashRAG corpus format
    
    Args:
        corpus_dir: Directory containing school JSON files
        output_file: Path to output corpus file
    
    Returns:
        Number of processed files
    """
    # Find all JSON files
    json_files = glob.glob(os.path.join(corpus_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {corpus_dir}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process each file and write to corpus
    processed_count = 0
    
    with open(output_file, "w") as f_out:
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, "r") as f_in:
                    school_data = json.load(f_in)
                    
                    # Extract basic information
                    school_name = school_data["basic_info"]["name"]
                    school_id = school_data["basic_info"]["school_id"]
                    location = school_data["basic_info"]["location"]["full_address"]
                    
                    # Extract metrics
                    metrics = ""
                    for category, values in school_data["metrics"].items():
                        if values:
                            metrics += f"\n{category}: " + ", ".join([f"{k}: {v}" for k, v in values.items()])
                    
                    # Format content
                    content = f"School: {school_name}\nLocation: {location}\n{metrics}"
                    
                    # Create document
                    doc = {"id": school_id, "contents": content}
                    
                    # Write to output file
                    f_out.write(json.dumps(doc) + "\n")
                    processed_count += 1
                    
                    # Print progress
                    if (i + 1) % 50 == 0 or i == len(json_files) - 1:
                        print(f"Processed {i+1}/{len(json_files)} files")
                        
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")
    
    return processed_count


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert NYC Schools data to FlashRAG corpus format')
    parser.add_argument('--corpus-dir', type=str, default='nyc_schools_data',
                        help='Directory containing school JSON files')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output corpus file path (default: <corpus_dir>/processed/corpus.jsonl)')
    
    args = parser.parse_args()
    
    # Process paths
    corpus_dir = args.corpus_dir
    if not os.path.isabs(corpus_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_dir = os.path.join(script_dir, corpus_dir)
    
    # Set default output path
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(corpus_dir, 'processed', 'corpus.jsonl')
    
    # Run conversion
    try:
        print(f"Starting conversion...")
        print(f"Corpus directory: {corpus_dir}")
        print(f"Output file: {output_file}")
        
        count = convert_school_data(corpus_dir, output_file)
        
        if count > 0:
            print(f"Conversion complete! Converted {count} school JSON files to corpus format: {output_file}")
            return 0
        else:
            print(f"Warning: No files processed. Please check input directory: {corpus_dir}")
            return 1
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 