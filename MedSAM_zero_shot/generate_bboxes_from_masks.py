# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 20:32:39 2025

@author: kisha
"""

import os
import cv2
import json
from tqdm import tqdm
# specifying where the masks masks are (which is what the boxes are based on)
mask_dir = "/content/drive/MyDrive/MedSAM/data/ctrus/masks"
output_json = "/content/drive/MyDrive/MedSAM/data/ctrus/bboxes.json"

annotations = [] #initialize to store the annothations

for filename in tqdm(os.listdir(mask_dir)): #looping thru all masks
    if not filename.endswith('.jpg'):
        continue

    mask_path = os.path.join(mask_dir, filename) 
    mask = cv2.imread(mask_path, 0) #grayscale 

    if mask is None:
        print(f"Skipped: {filename}") #in case there were issues when downloading the images
        continue

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) #needed to convert a grayscale image to a binary mask for the boxes (255 as white)
    if not (binary_mask > 0).any():
        print(f"No mask in: {filename}") #to avoid image pairs with no masks
        continue

    x, y, w, h = cv2.boundingRect((binary_mask > 0).astype('uint8'))
    annotations.append({
        "image": filename,
        "bbox": [x, y, x + w, y + h] #specifying format for boxes (led to issues in intial runs)
    })

# making sure output is saved in proper folder
with open(output_json, "w") as f: 
    json.dump(annotations, f, indent=2)

print(f"Saved {len(annotations)} bounding boxes to {output_json}")
