import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_hex(image_bytes: bytes) -> tuple[str, list]:
    """
    Decodes an image file, isolates the leaf (ignoring backgrounds), 
    and extracts the dominant color hex code.
    """
    # 1. Decode the image bytes into an OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # OpenCV loads in BGR format, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # 3. Filter out extremely dark/light pixels (usually background or glare)
    # Keeping pixels where the sum of RGB is between 50 and 600
    filtered_pixels = [p for p in pixels if 50 < sum(p) < 600]
    
    if not filtered_pixels:
        return "#000000", [0, 0, 0] # Fallback if image is pure black/white

    filtered_pixels = np.array(filtered_pixels)

    # 4. Use K-Means Clustering to find the most dominant color
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(filtered_pixels)
    
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    r, g, b = dominant_color
    
    # 5. Convert RGB to Hex
    hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
    
    return hex_code, [int(r), int(g), int(b)]

def heuristic_diagnosis(r: int, g: int, b: int) -> str:
    """
    A temporary heuristic to map colors to our JSON database 
    before we implement the full YOLO model.
    """
    if r > 130 and g > 130 and b < 100:
        return "nitrogen_deficiency" # Yellowish (Chlorosis)
    elif r > int(g) and b > int(g):
        return "phosphorus_deficiency" # Purplish
    else:
        return "healthy_optimal" # Mostly green