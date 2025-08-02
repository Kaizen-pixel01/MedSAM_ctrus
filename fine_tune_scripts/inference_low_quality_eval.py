# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 16:08:07 2025

@author: kisha
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import glob
import json
from segment_anything import sam_model_registry

join = os.path.join

# Dataset
class NpyDataset(Dataset):
    def __init__(self, data_root): #same set up format as training script (making sure paths and folders match up)
        self.img_path = join(data_root, "imgs")
        self.gt_path = join(data_root, "gts")
        self.img_files = sorted(glob.glob(join(self.img_path, "*.npy"))) #sorted list of the npy files
        self.gt_files = sorted(glob.glob(join(self.gt_path, "*.npy")))
        self.img_files = [f for f in self.img_files if os.path.basename(f) in [os.path.basename(g) for g in self.gt_files]] #only need image pairs
        print(f" Found {len(self.img_files)} image–mask pairs.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): #loading image and keeping same format as previous tests
        img_name = os.path.basename(self.img_files[idx])
        img = np.load(os.path.join(self.img_path, img_name))  # (H, W, 3)
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        #ground truth
        mask = np.load(os.path.join(self.gt_path, img_name))  # (H, W)
        label = (mask > 0).astype(np.uint8)
        #getting bounding box 
        y_idx, x_idx = np.where(label > 0)
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        box = np.array([x_min, y_min, x_max, y_max])
        #getting all needed vals - image, box, mask and gfilename
        return (
            torch.tensor(img).float(),
            torch.tensor(label[None, :, :]).long(),
            torch.tensor(box).float(),
            img_name
        )

# Model 
class MedSAM(nn.Module): #ensuring pretrained componets are being used - freezing prompt encoder same as training
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder   #extracting the featres
        self.mask_decoder = mask_decoder    #getting the input prompts (which only be box for my experiments) into features
        self.prompt_encoder = prompt_encoder  #generating the masks (using the features obtained)
        for param in self.prompt_encoder.parameters(): #the MedSAM authors recommended this so it was probably to avoind issues when training on new images (i.e overfitting)
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image) #image embedding
        #box prompts
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  #needs to match shape of prompt encoder
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=box_torch, masks=None) #as mentioned earlier I'm only using box for the experiments (so the rest can be set to none)
        #making segmentations - calling it low res masks since it is not the same as input image so it needs to be matched at the end
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        #matching resolution - similar to resizing previously
        ori_res_masks = F.interpolate(low_res_masks, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return ori_res_masks

#  Metrics 
def compute_metrics(preds, targets): #same format as training script - needed for comparisons amongst models
    smooth = 1e-6
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    iou_denom = (preds + targets - preds * targets).sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (iou_denom + smooth)
    precision = (intersection + smooth) / (preds.sum(dim=1) + smooth)
    recall = (intersection + smooth) / (targets.sum(dim=1) + smooth)

    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
    }

#  Main 
def main(): #this way i can parse commands to run different tests with different fine-tuned MedSAM models 
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-model_path", type=str, required=True)
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    #makig sure to save the evaluation results
    save_dir = join("/content/drive/MyDrive/MedSAM/eval_results", "Eval-" + datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(save_dir, exist_ok=True)
    #loading the model - same as before
    device = torch.device(args.device)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.model_path)
    model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    #loas the dataset
    dataset = NpyDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
    count = 0
    #the actual inference loop foing through the images
    for image, mask, box, name in tqdm(loader):
        image, mask = image.to(device), mask.to(device)
        box_np = box.cpu().numpy()
        with torch.no_grad(): #same as training script
            pred = model(image, box_np)
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            mask_resized = F.interpolate(mask.float(), size=pred.shape[2:], mode="nearest")

            metrics = compute_metrics(pred_binary, mask_resized)
            for k in metrics:
                metrics_sum[k] += metrics[k]
            count += 1
            #saving visualization for first 4 samples (for visual analysis and can potentially add to the poster)
            if count <= 4:
                img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                gt_np = mask_resized[0][0].cpu().numpy()
                pred_np = pred_binary[0][0].cpu().numpy()
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img_np.astype(np.uint8))
                axs[0].set_title("Image")
                axs[1].imshow(gt_np, cmap='gray')
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred_np, cmap='gray')
                axs[2].set_title("Prediction")
                for ax in axs: ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{count}.png"))
                plt.close()

    avg_metrics = {k: metrics_sum[k] / count for k in metrics_sum}
    print("Evaluation Results:", avg_metrics)

    # Saving the metrics
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f_txt, open(os.path.join(save_dir, "metrics.json"), "w") as f_json:
        for k, v in avg_metrics.items():
            f_txt.write(f"{k}: {v:.4f}\n")
        json.dump(avg_metrics, f_json, indent=4) #makes it wasy to extract for analysis later

    # Plotting bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(avg_metrics.keys(), avg_metrics.values(), color='skyblue')
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    for i, (k, v) in enumerate(avg_metrics.items()):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_bar_chart.png"))
    plt.close()

if __name__ == "__main__":
    main()
