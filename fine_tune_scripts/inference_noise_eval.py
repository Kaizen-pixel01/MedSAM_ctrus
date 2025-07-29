# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:39:37 2025

@author: kisha
"""

import os
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
from segment_anything import sam_model_registry

join = os.path.join

# --------------------- Speckle Noise --------
#adding  noise to an image to stimulate speckle noise in real world practice
def add_speckle_noise(image, variance=0.1): #variance is what determines the level of noise added to the image
    noise = np.random.normal(0, np.sqrt(variance), image.shape)
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# ---------------------- Dataset --------
class NpyDataset(Dataset): #making sure to follow the same naming format for making the .npy conversions
    def __init__(self, data_root, apply_noise=False, noise_variance=0.1):
        self.img_path = join(data_root, "imgs")
        self.gt_path = join(data_root, "gts")
        self.img_files = sorted(glob.glob(join(self.img_path, "*.npy")))
        self.gt_files = sorted(glob.glob(join(self.gt_path, "*.npy")))
        #making sure ther is a pair to avoid errors
        self.img_files = [f for f in self.img_files if os.path.basename(f) in [os.path.basename(g) for g in self.gt_files]]
        self.apply_noise = apply_noise
        self.noise_variance = noise_variance
        print(f"Found {len(self.img_files)} image–mask pairs.") #making sure it matches up with EDA

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): #loading the images and adding the noise 
        img_name = os.path.basename(self.img_files[idx])
        img = np.load(os.path.join(self.img_path, img_name))  # (H, W, 3)
        if self.apply_noise: #added to make it so that the same test can be run again without the noise added to ensure the test is not swayed due to technical error (using different scripts)
            img = add_speckle_noise(img, self.noise_variance)
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        #loading the binary mask
        mask = np.load(os.path.join(self.gt_path, img_name))  # (H, W)
        label = (mask > 0).astype(np.uint8)
        #calculating bounding boxes
        y_idx, x_idx = np.where(label > 0)
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        box = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img).float(),
            torch.tensor(label[None, :, :]).long(),
            torch.tensor(box).float(),
            img_name
        )

# ---------------------- MedSAM Model --------
class MedSAM(nn.Module): #same as before - referencing repo
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters(): #freezing prompt encoder since that is not something i want to change in training
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad(): #same as before - making the prompts using no points
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        #same resizing - to make sure preduction matches inputs
        ori_res_masks = F.interpolate(low_res_masks, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return ori_res_masks

# ----------------------- Metrics --------
def compute_metrics(preds, targets): #same across all tests
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

# --------------------- Main ----------
def main(): #same case of allowing for arguements to be custom so i can use it with different testing parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-model_path", type=str, required=True)
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_variance", type=float, default=0.1) #here is where noise level can be changed (valriance)
    args = parser.parse_args()

    tag = "Noisy" if args.add_noise else "Clean" #to help differentiate the results folder to keep organized (initial runs were hard to track so needed to add this)
    save_dir = join("/content/drive/MyDrive/MedSAM/eval_results", f"{tag}-Eval-" + datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(args.device)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.model_path)
    model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    dataset = NpyDataset(args.data_path, apply_noise=args.add_noise, noise_variance=args.noise_variance)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0} #initialize metrics
    count = 0
    #inference and evaluation - looping through the given input images
    for image, mask, box, name in tqdm(loader):
        image, mask = image.to(device), mask.to(device)
        box_np = box.cpu().numpy()
        with torch.no_grad():
            pred = model(image, box_np)
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            mask_resized = F.interpolate(mask.float(), size=pred.shape[2:], mode="nearest") #resizing to make sure ground truth matches so that the plots are actually usable (first few were unusable since there was no resizing)

            metrics = compute_metrics(pred_binary, mask_resized) 
            for k in metrics:
                metrics_sum[k] += metrics[k]
            count += 1

            if count <= 5:
                img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                gt_np = mask_resized[0][0].cpu().numpy()
                pred_np = pred_binary[0][0].cpu().numpy()
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img_np.astype(np.uint8))
                axs[0].set_title("Image (Noisy)" if args.add_noise else "Image")
                axs[1].imshow(gt_np, cmap='gray')
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred_np, cmap='gray')
                axs[2].set_title("Prediction")
                for ax in axs: ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{count}.png"))
                plt.close()
# Summary of model performance
    avg_metrics = {k: metrics_sum[k] / count for k in metrics_sum}
    print("Evaluation Results:", avg_metrics)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()
