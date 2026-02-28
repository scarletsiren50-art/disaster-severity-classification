import os
import json
import cv2
import numpy as np

RAW_IMAGE_DIR = "data/raw/xview2/train/images"
RAW_LABEL_DIR = "data/raw/xview2/train/labels"
OUTPUT_DIR = "data/processed"

IMG_SIZE = 128
MAX_PER_CLASS = 300  # keep dataset small for your laptop

damage_map = {
    "no-damage": "no_damage",
    "minor-damage": "minor_damage",
    "major-damage": "major_damage",
    "destroyed": "destroyed"
}

def create_output_folders():
    for folder in damage_map.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

def parse_wkt_polygon(wkt_string):
    # Extract coordinates from WKT string
    coords_text = wkt_string.replace("POLYGON ((", "").replace("))", "")
    coords_pairs = coords_text.split(",")
    coords = []

    for pair in coords_pairs:
        x, y = pair.strip().split()
        coords.append([float(x), float(y)])

    return np.array(coords, dtype=np.int32)


def extract_buildings():
    create_output_folders()

    class_count = {k: 0 for k in damage_map.values()}

    label_files = os.listdir(RAW_LABEL_DIR)

    for label_file in label_files:

        if not label_file.endswith("_post_disaster.json"):
            continue  # only post-disaster has damage

        label_path = os.path.join(RAW_LABEL_DIR, label_file)

        with open(label_path) as f:
            data = json.load(f)

        image_name = label_file.replace(".json", ".png")
        image_path = os.path.join(RAW_IMAGE_DIR, image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        if "features" not in data or "xy" not in data["features"]:
            continue

        for feature in data["features"]["xy"]:

            damage = feature["properties"].get("subtype", None)

            if damage not in damage_map:
                continue

            class_name = damage_map[damage]

            if class_count[class_name] >= MAX_PER_CLASS:
                continue

            wkt_string = feature.get("wkt", None)
            if wkt_string is None:
                continue

            coords = parse_wkt_polygon(wkt_string)

            x, y, w, h = cv2.boundingRect(coords)
            crop = image[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            save_path = os.path.join(
                OUTPUT_DIR,
                class_name,
                f"{label_file}_{class_count[class_name]}.png"
            )

            cv2.imwrite(save_path, crop)

            class_count[class_name] += 1

    print("Extraction complete.")
    print(class_count)