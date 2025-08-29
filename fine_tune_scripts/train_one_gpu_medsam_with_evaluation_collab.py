# -*- coding: utf-8 -*-
"""

@author: kisha
"""

# train_one_gpu.py - reference file from repo (since most of the structure was there, i just had to add my pathing and also had to change some parts,removing functions that were not useful as well, to make it work with c-trus)
# training MedSAM on npy image-mask pairs 

import numpy as np
import os
import sys
sys.path.append("C:/Users/kisha/MedSAM")
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything import sam_model_registry
import argparse
import monai
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

join = os.path.join

#  Dataset 
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "*.npy")))
        self.img_path_files = sorted(glob.glob(join(self.img_path, "*.npy")))
        self.gt_path_files = [
            f for f in self.gt_path_files if os.path.basename(f) in [os.path.basename(p) for p in self.img_path_files]
        ]
        self.bbox_shift = bbox_shift
        print(f" Number of image-mask pairs: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(os.path.join(self.img_path, img_name))  # (H, W, 3)
        img = np.transpose(img, (2, 0, 1))  # Convert to (3, H, W)

        mask = np.load(os.path.join(self.gt_path, img_name))  # (H, W)
        label_ids = np.unique(mask)[1:]  # # ignore background label 0 - kept throwing attribute errors

        if len(label_ids) == 0: #had issues with normalization since it would constantly run into issues so i used raw .npy 
            label = np.zeros_like(mask)
        else:
            label = (mask == random.choice(label_ids)).astype(np.uint8)

        y_indices, x_indices = np.where(label > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = label.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        box = np.array([x_min, y_min, x_max, y_max])
        #made it simpler as original training file had more options (ie. 2D and so on)
        return (
            torch.tensor(img).float(),
            torch.tensor(label[None, :, :]).long(),
            torch.tensor(box).float(),
            img_name,
        )

#  MedSAM formating (kept almost identical to reference file - also freezing the encoder as recommended by github repo)
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
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
        #needed to resuze masks to make sure that the predictions masked the input sizes 
        ori_res_masks = F.interpolate(low_res_masks, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return ori_res_masks

#  Metrics 
def compute_metrics(preds, targets): #custom function added different from reference file since I needed the results to match other test formatso i could compare peformance between the models
    smooth = 1e-8 #same as with zero-shot - to avoid 0 errors
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

#  Main Training 
def main(): #kept arg together since the reference kept it split up but i found it hard to follow so i put all in the same place (also easier to change when doing different types of training)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tr_npy_path", type=str, default="C:/Users/kisha/MedSAM/data/npy_sample") #local file that i tested on when first testing training
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="C:/Users/kisha/MedSAM/work_dir/MedSAM/medsam_vit_b.pth") #downloaded checkpoint file to test locally
    parser.add_argument("--load_pretrain", type=bool, default=True)
    parser.add_argument("-num_epochs", type=int, default=3)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu") #initially tested on anaconda but due to hardware limitations i switched to collab
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join("/content/drive/MyDrive/MedSAM/work_dir", args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)

    device = torch.device(args.device)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder).to(device)
    model.train()

    train_dataset = NpyDataset(args.tr_npy_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    best_loss = 1e10
    losses = []

    for epoch in range(args.num_epochs):
        total_loss = 0
        metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
        count = 0

        for image, mask, box, _ in tqdm(train_loader):
            image, mask = image.to(device), mask.to(device)
            box_np = box.detach().cpu().numpy()
            pred = model(image, box_np)

            mask_resized = F.interpolate(mask.float(), size=pred.shape[2:], mode="nearest")

            loss = seg_loss(pred, mask_resized) + ce_loss(pred, mask_resized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad(): 
                pred_sigmoid = torch.sigmoid(pred) #binary format for predictions 
                pred_binary = (pred_sigmoid > 0.5).float()
                metrics = compute_metrics(pred_binary, mask_resized)
                for k in metrics:
                    metrics_sum[k] += metrics[k]
                count += 1

        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: metrics_sum[k] / count for k in metrics_sum}
        losses.append(avg_loss)

        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        print(f"Epoch {epoch} Metrics -> Dice: {avg_metrics['dice']:.4f}, IoU: {avg_metrics['iou']:.4f}, "
              f"Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}")

        # Save visualization - added it to provide visual outputs at the end of epochs so that i can make qualitative observatons
        model.eval()
        with torch.no_grad():
            image_vis, mask_vis, box_vis, _ = next(iter(train_loader))
            image_vis, mask_vis = image_vis.to(device), mask_vis.to(device)
            box_np_vis = box_vis.detach().cpu().numpy()
            pred_vis = model(image_vis, box_np_vis)
            pred_vis_sigmoid = torch.sigmoid(pred_vis)
            pred_binary = (pred_vis_sigmoid > 0.5).float()
            mask_vis_up = F.interpolate(mask_vis.float(), size=pred_binary.shape[2:], mode="nearest")

            img_np = image_vis[0].cpu().numpy().transpose(1, 2, 0)
            gt_np = mask_vis_up[0][0].cpu().numpy()
            pred_np = pred_binary[0][0].cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_np.astype(np.uint8))
            axs[0].set_title("Input Image")
            axs[1].imshow(gt_np, cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(pred_np, cmap='gray')
            axs[2].set_title("Prediction")
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(model_save_path, f"epoch_{epoch}_viz.png"))
            plt.close()
        model.train()

        torch.save(model.state_dict(), join(model_save_path, "medsam_model_latest.pth"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), join(model_save_path, "medsam_model_best.pth"))

    # Plotting training loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(model_save_path, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()





