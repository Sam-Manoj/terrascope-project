import os
import csv
from pathlib import Path
from chromatic_extractor import extract_dominant_hex

def generate_csv_dataset():
    # 1. Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = base_dir / "datasets" / "raw"
    csv_output_path = base_dir / "datasets" / "leaf_features.csv"
    
    # The folders we want to scan (these match your JSON database keys)
    classes = ["nitrogen_deficiency", "phosphorus_deficiency", "healthy_optimal"]
    
    # 2. Open the CSV file to write
    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["red", "green", "blue", "label"])
        
        print(f"Starting extraction. Scanning {raw_data_dir}...\n")
        
        # 3. Loop through each class folder
        for class_name in classes:
            folder_path = raw_data_dir / class_name
            
            if not folder_path.exists():
                print(f"Skipping {class_name}... folder not found.")
                continue
                
            # 4. Process every image in the folder
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = folder_path / image_file
                    
                    try:
                        # Read the file as bytes (simulating how the API receives it)
                        with open(img_path, "rb") as f:
                            image_bytes = f.read()
                        
                        # Extract the RGB values using our existing function
                        hex_code, rgb = extract_dominant_hex(image_bytes)
                        r, g, b = rgb
                        
                        # Write the row to the CSV
                        writer.writerow([r, g, b, class_name])
                        print(f"Processed: {image_file} -> {hex_code}")
                        
                    except Exception as e:
                        print(f"Failed to process {image_file}: {e}")
                        
    print(f"\nDataset generation complete! Saved to {csv_output_path}")

if __name__ == "__main__":
    generate_csv_dataset()