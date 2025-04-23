import os
import glob
import json
import argparse
from collections import Counter

def collect_styles(labels_root):
    """Parse all JSON files to collect basestyle values."""
    styles = []

    json_files = glob.glob(os.path.join(labels_root, "*", "*", "*.json"))
    print(f"Found {len(json_files)} JSON files.\n")

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping corrupt JSON: {json_path}")
            continue

        style = data.get("basestyle")
        if style:
            styles.append(style)

    style_counter = Counter(styles)
    print("Unique styles found (sorted alphabetically):\n")
    for style, count in sorted(style_counter.items()):
        print(f"{style}: {count}")

    style_list = sorted(style_counter.keys())
    print("\n\nSTYLE_CLASSES = [")
    for s in style_list:
        print(f'    "{s}",')
    print("]")

    return style_list

def main():
    parser = argparse.ArgumentParser(description="Collect unique hair styles from JSON label files.")
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels directory')

    args = parser.parse_args()
    collect_styles(args.labels_path)

if __name__ == "__main__":
    main()
