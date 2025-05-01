import os
import json
from collections import defaultdict

# Paths
json_root = "16726-finalproject/hair_json"
image_root = "16726-finalproject/hair_images"

# Track counts and example image paths
style_counts = defaultdict(int)
style_examples = {}

# Traverse JSON directory
for dirpath, _, filenames in os.walk(json_root):
    for file in filenames:
        if file.endswith(".json"):
            json_path = os.path.join(dirpath, file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                basestyle = data.get("basestyle")
                image_path = data.get("path")

                # Normalize and reconstruct full image path
                if basestyle and image_path:
                    # Extract the final part of the image path
                    filename = os.path.basename(image_path)
                    
                    # Get relative folder path from JSON path
                    rel_dir = os.path.relpath(dirpath, json_root)
                    
                    # Construct actual image path
                    actual_image_path = os.path.join(image_root, rel_dir, filename)

                    style_counts[basestyle] += 1
                    
                    # Save first encountered image path as representative
                    if basestyle not in style_examples:
                        style_examples[basestyle] = actual_image_path
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

# Print sorted results
sorted_styles = sorted(style_counts.items(), key=lambda x: -x[1])
for style, count in sorted_styles:
    example_path = style_examples[style]
    print(f"{style} | {count} | {example_path}")
