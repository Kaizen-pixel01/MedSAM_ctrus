# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 20:31:02 2025

@author: kisha
"""


import os
import cv2
import numpy as np
from tqdm import tqdm

# File pats 
gt_mask_dir = "/content/drive/MyDrive/MedSAM/data/ctrus/masks"
pred_mask_dir = "/content/drive/MyDrive/MedSAM/data/ctrus/predicted_masks"

#  Helper functionsto compute dice and iou scores
def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-8)

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-8)

# Getting all predictions 
pred_files = [f for f in os.listdir(pred_mask_dir) if f.endswith("_mask.png")]
# initialize the metrics
dice_scores = []
iou_scores = []
skipped = 0

for pred_file in tqdm(pred_files): #iterating through the preducted maskes made from previous inference file 
    base_name = pred_file.replace("_mask.png", ".jpg")
    pred_path = os.path.join(pred_mask_dir, pred_file) #predicted mask
    gt_path = os.path.join(gt_mask_dir, base_name) #gt mask

    if not os.path.exists(gt_path): #skipping when no gt mask (debugging as well)
        skipped += 1
        continue

    # loading in grayscale
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred_mask is None or gt_mask is None:
        skipped += 1
        continue
    # Load masks and make binary - so converting the pixels based on the threshold same as bbox file
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    #calculating the performance metrics
    dice = dice_score(gt_binary, pred_binary)
    iou = iou_score(gt_binary, pred_binary)
    #store the results
    dice_scores.append(dice)
    iou_scores.append(iou)

# Summarizing the results 
if dice_scores:
    print(f"\n Evaluated {len(dice_scores)} image(s)")
    print(f" Dice Score: Mean = {np.mean(dice_scores):.4f}, Std = {np.std(dice_scores):.4f}")
    print(f" IoU Score:  Mean = {np.mean(iou_scores):.4f}, Std = {np.std(iou_scores):.4f}")
else:
    print(" No valid predictions evaluated. Check filenames and mask paths.")

if skipped > 0:
    print(f" Skipped {skipped} image(s) due to missing files or read errors.")
