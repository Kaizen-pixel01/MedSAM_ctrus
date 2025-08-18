# -*- coding: utf-8 -*-
"""

@author: kisha
"""


import os
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

#  File Paths 
image_dir = "/content/drive/MyDrive/MedSAM/data/ctrus/images"
bbox_json = "/content/drive/MyDrive/MedSAM/data/ctrus/bboxes.json"
checkpoint = "/content/drive/MyDrive/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
output_dir = "/content/drive/MyDrive/MedSAM/data/ctrus/predicted_masks"

os.makedirs(output_dir, exist_ok=True) #helps to make sure that the needed paths were made and named properly (also helps for when you need to debug errors)

# Load bounding boxes - different format from the CLI used in medsam inference file since it kept running errors with C-trus images
with open(bbox_json, "r") as f:
    bboxes = json.load(f)
bbox_dict = {entry["image"]: entry["bbox"] for entry in bboxes}

# Load MedSAM Model - very similar methodology to inference file with single image
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=None) #using MedSAM pretrained checkpoint as provided in gitub repo
state_dict = torch.load(checkpoint, map_location=torch.device("cpu")) 
sam.load_state_dict(state_dict)  #loads all the necessary weights as well since that wont be change d in zero shot
sam.to(device)

predictor = SamPredictor(sam) #needed for prompt inference (i.e image loading, prompt processing, etc)
'''
had troubles adapting MedSAm repo version since i didn't use CLI for the prompt boxes like with the example so I had to refer to SAM repo to use alternative function
'''

processed = 0
skipped = 0
#Limit to 10 valid images for testing if the model works (can remove after making sure it works - so just getting rif of the limit)
# limit = 10
count = 0

# Only loop over the images that are known have bounding boxes
for filename in list(bbox_dict.keys()):

    image_path = os.path.join(image_dir, filename) #loading image
    
    #for debugging since the first download got interupted and it saved empty files
    if not os.path.exists(image_path):
        print(f" Skipping (image file not found): {filename}")
        skipped += 1
        continue
    #same thought process - avoid any kind of daulty files
    image = cv2.imread(image_path)
    if image is None:
        print(f" Skipping (cannot read image): {filename}")
        skipped += 1
        continue

    #convert grayscale to RGB when needed - for MedSAM input 
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_box = np.array([bbox_dict[filename]]) #MedSAM expects an array 

    try: #to prevent run from craching since it was an issue (I also added a skip so that I know when an issue is occuring)
        predictor.set_image(image) #sets image
        transformed_box = predictor.transform.apply_boxes_torch(
            torch.tensor(input_box, dtype=torch.float32, device=device),
            image.shape[:2] #transformations needed to match others
        )
        
        #zero shot so no prompt points
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            multimask_output=False #only need one mask
        )
        
        #saving the prediction
        pred_mask = (masks[0][0].cpu().numpy() * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, filename.replace(".jpg", "_mask.png"))
        cv2.imwrite(out_path, pred_mask)
        #keep count 
        processed += 1
        count += 1

    except Exception as e: #for debugging since initial inference kept having issues
        print(f" Error processing {filename}: {e}")
        skipped += 1

# final Summary 
print(f"\n DONE FINALLY GOT IT TO WORK!")
print(f"Total processed: {processed}")
print(f"Total skipped: {skipped}")

