# -*- coding: utf-8 -*-
"""

@author: kisha

"""
"""
wilcoxon signed-rank comparisons for per-image metrics(need to be paried for evaluatiom to work)

inputs are the two CSVs with columns for the four metrics
outputs a txt and csv which shows the results of the evaluation (can use as table or just for quick analysis check)
"""

# liberaries needed for the test
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

METRICS = ["dice", "iou", "precision", "recall"] #list format that will define the performance metrics being compared

def main():  # i included arguement for key so that this can be used with experiment for patients as well - something to look at for future studies and tests so that it replicates the c-trus benchmark study
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", required=True)
    ap.add_argument("--csv_b", required=True)
    ap.add_argument("--label_a", default="A") #the two models being compared labelled as A and B
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--image_key", default="image")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--alpha", type=float, default=0.05) #significance level (normaly 0.05 for significance)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) #can make custom path to organize the results so that its not confusing since I use this multiple times in the analysis
    out_dir.mkdir(parents=True, exist_ok=True) #aalways making sure the paths exist otherwise it will lead to headache errors 
    # read in both csv metrics files that are bieng compared with args as well
    a = pd.read_csv(args.csv_a)
    b = pd.read_csv(args.csv_b)


    # inner-join on key column (which will be the image names) to get paired rows between the sets of diffrences
    merged = a[[args.image_key] + METRICS].merge(
        b[[args.image_key] + METRICS],
        on=args.image_key,
        suffixes=(f"_{args.label_a}", f"_{args.label_b}") #keeping track of what the metric belongs to
    )
    n_pairs = len(merged)
    if n_pairs == 0: #to make sure that pairs are compared (i originally had issues with naming results from exp3 and 4 so got error when trying to compare results for both - this way i knew to check which results I am using)
        raise SystemExit(
            "No overlapping images between the two CSVs. " #letting me know to check if I'm using the correct files
        )
    #initializing for txt and csv file for results of the test
    rows = []
    lines = []
    lines.append(f"Wilcoxon signed-rank (two-sided) | pairs: n={n_pairs}") #lets me double check that the correct experiment is being tested (28 for testing on LQ test set and 508 for entire c-trus test)
    lines.append(f"Condition A: {args.label_a}  |  Condition B: {args.label_b}") #showing the two models compared
    lines.append("")

    for m in METRICS: #for each metric takes values from each test so that the difference can be calculated
        xa = merged[f"{m}_{args.label_a}"].astype(float).to_numpy() #ensure that the valures stay as arrays for calculation
        xb = merged[f"{m}_{args.label_b}"].astype(float).to_numpy()

       
        # do wilcoxon test (zero_method='wilcox' ignores zero differences) and two-sided refers to which is better than the other (not just looking at 1 as greater or less - the other two options for Wilcoxon) 
        stat, p = stats.wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided") #gives the test results which cna be used to detemine if the differences are significant based on the median of the paired differences
        #calculating mean and std for bith as well (this will make it easier to display differenes between experiments as well)
        mean_a, sd_a = float(np.mean(xa)), float(np.std(xa, ddof=1)) if len(xa) > 1 else float("NA") #to make sure that there is more than 1 data point so it doesnt give error
        mean_b, sd_b = float(np.mean(xb)), float(np.std(xb, ddof=1)) if len(xb) > 1 else float("NA") #same case here
        
        significant = p < args.alpha #to let me know if the differences are significant or not
        # So the results from this test will drop 0 difference pairs and rank  the absolute differences which is sumed as W(Wilcoxon signed-rank statistic), the  p values is treated the same as other significance tests refering to the possibility of chance
       
        #dictionary with all the results to make into csv file if needed for further analysis (also provides a table that can be used for results as well)
        rows.append({
            "metric": m,
            "n_pairs": n_pairs,
            f"mean_{args.label_a}": mean_a,
            f"sd_{args.label_a}": sd_a,
            f"mean_{args.label_b}": mean_b,
            f"sd_{args.label_b}": sd_b,
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "significant_at_alpha": significant,
        })
        
        #making sure to output the important results in a readable format so that it can be checked easily
        lines.append(
            f"[{m.upper()}] " #capitalized the metrics to stand out
            f"mean_{args.label_a}={mean_a:.4f}±{sd_a:.4f} vs "
            f"mean_{args.label_b}={mean_b:.4f}±{sd_b:.4f}  |  "
            f"W={stat:.4f}, p={p:.4g}  "
            f"{'**SIGNIFICANT**' if significant else '(ns)'}" # main takeaway from this test
        )

    # saving both the csv and txt files for analysis
    out_csv = out_dir / "wilcoxon_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = out_dir / "wilcoxon_report.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    #to help keep track where the files are     
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
   

if __name__ == "__main__":
    main()
