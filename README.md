# MedSAM C-TRUS Segmentation Repository

This repository contains scripts and notebooks for evaluating and fine-tuning **MedSAM** on the **C-TRUS** dataset, which consists of grayscale ultrasound images for colon wall segmentation.

The project is organized into two main folders:

---

## MedSAM_zero_shot

This folder includes all the files necessary to perform **zero-shot inference** using MedSAM on the C-TRUS dataset.

- Python scripts for preprocessing, bounding box generation, inference, and evaluation
- A Jupyter Notebook that ties these components together to run the complete zero-shot pipeline

### Key Features:
- Uses pre-trained MedSAM (vit_b) checkpoint
- Inference based on bounding box prompts
- Evaluation metrics: Dice Score, IoU, Precision, Recall

---

## fine_tune_scripts

This folder contains scripts and a notebook to **fine-tune** MedSAM on the C-TRUS dataset and run additional evaluation experiments.

- Training routines using .py files adapted from the official MedSAM GitHub repository
- Evaluation scripts for:
  - Performance on low-quality images
  - Noise robustness testing
  - Generalization tests on different training-validation splits


### Key Features:
- Supports training from scratch or continuing from a pre-trained checkpoint
- Computes standard segmentation metrics
- Visualizes predictions and saves evaluation results
- Can simulate real-world conditions with noisy or low-quality inputs


---
### The next few folders help tie the project together

---
## Files for training 
 This folder containes the filtered csv file for the C-TRUS dataset that is used in the training step of the fine-tuned analysis. In order to make the code work, you would need to download the file into the appropriate directory to match the path in the Jupyter notebook for MedSAM training. 

---

## analysis and visuals
  This folder contains the files and noteooks required to perform statistical analysis on the results from the experiments as well as how I used the results to create visuals. 
  
---
## EDA

This folder contains a npotebook that provides an  **eploratory data analysis (EDA)** of the C-TRUS dataset.

- This helps get a better understanding of the C-TRUS dataset as well as some preliminary analysis that can be made on it

### Key Features:
- **Image Quality Distribution**: Visual breakdown of high vs. low-quality ultrasound images (based on metadata annotations).
- **Mask Coverage**: Computes and visualizes the pixel-wise area covered by the colon wall masks.
- **Bounding Box Statistics**: Analyzes dimensions of regions of interest.
- **Class Balance**: Reports how many samples are annotated as "low-quality" vs "high-quality".
- **Sample Visualizations**: Displays example ultrasound images along with their segmentation masks for qualitative insight.

---

## Requirements

To run the code, you should have the MedSAM environment set up as per [MedSAM GitHub instructions](https://github.com/bowang-lab/MedSAM).
- You can clone the MedSAM github repository so that you can have all the necessary files used in the scripts
In order to be able to run the training file without changing any details you will have to do the followng:
- Make sure that the files are placed in the proper location since I have used Collab, the file paths used are for GoogleDrive. This means that in order to get the training to work you will have to be wary of the following files and make sure they are placed in the right spot. The main pathing will be /content/drive/MyDrive/MedSAM (" in the following examples) and the rest will depend on the files being used. The following are the files you need to look out for:
  - c-trus.filtered.csv should be placed  "/data/ctrus/
  - train_one_gpu_medsam_with_evaluation_collab.py should be placed in "/fine_tune_scripts/
  - inference_low_quality_eval_per.py should be placed in "/fine_tune_scripts/
  - inference_noise_eval_per.py should be placed in "/fine_tune_scripts/
- For zero-shot the files should be placed in a different directory: /content/drive/MyDrive/MedSAM/zero_shot_result_scripts
  - generate_bboxes_from_masks.py
  - batch_medsam_ctrus_inference.py
  - batch_medsam_ctrus_inference_final.py
  - evaluate_medsam_predictions.py
- Also since zeroshot uses the masks from C-TRUS dataset, make sure that you have downloaded the MedSAM repo and place it in /content/drive/MyDrive/ if you want to make no changes to the code. Otherwise you would have to update the generate_bboxes_from_masks.py with your correct directory that you saved MedSAM in.
- I have also provided the three MedSAM checkpoint folder names here. These are a result of training the MedSAM model using a given subset. The issue is that these folders are too large to download from Goog Drive, so I have included the names here as a reminder to change your trained checkpoints to match these names when evaluating. Another option is to use the correct pathing for your own checkpoint forlder when running the test (since the inference files let you change checkpoint paths). If you want to the same pathing as me, the following folders should be placed in /content/drive/MyDrive/MedSAM/work_dir/
  - MedSAM-ViT-B-20250726-high
  - MedSAM-ViT-B-20250728-mixedS
  - MedSAM-ViT-B-20250728-lowS

---

##  Dataset
This project uses the [C-TRUS dataset](https://github.com/wwu-mmll/c-trus), which provides annotated transabdominal ultrasound images for colon wall segmentation tasks.

The C-TRUS dataset should be preprocessed into a format compatible with the scripts:
- Images: .npy or .jpg format
- Masks: ground truth segmentations
- Bounding boxes: JSON file or generated from masks
- Also filtered metadata file to include only image-mask pairs that had valid ground truth masks (where the mask area was not 0) and it was called ctrus.filtered.csv (and it is used when testing MedSAM)

---

##  Imprtant Changes to Original Medsam Reference Files

For this project, I had to make some adjustments to the original MedSAM code so that it would work with the C-TRUS dataset. I ran zero-shot tests with box prompts, saving the results, and keeping track of performance for each image. Since the reference code didn’t provide detailed evaluation, I also created my own evaluation step to calculate metrics like Dice, IoU, precision, and recall. I did this by binarizing the predicted masks (which were just soft masks produced by MedSAM) using thresholds. This way I could use it alongside the ground truth mask (which I also binzarized in the preprocessing step that converts the original .jpg images into .npy arrays) to calculate overlap. I also tweaked the reference training script to better all fine-tuning with different image subsets and saving the progress by updating the checkpoint file. 





