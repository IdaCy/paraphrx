#!/usr/bin/env python3
"""
python e_eval/src/sort_status.py \
    logs/id_status_phrxing50k_part5b.json \
    logs/id_status_phrxing50k_part5b_sorted.json
"""
import json
from collections import OrderedDict
import sys

def sort_json_keys_numeric(input_path, output_path):
    # Load the JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Create a new OrderedDict, sorting keys by their integer value
    sorted_data = OrderedDict(
        (k, data[k]) for k in sorted(data.keys(), key=lambda x: int(x))
    )
    # Write it back out with the same formatting style
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # newline at end of file

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input.json> <output.json>')
        sys.exit(1)
    sort_json_keys_numeric(sys.argv[1], sys.argv[2])
