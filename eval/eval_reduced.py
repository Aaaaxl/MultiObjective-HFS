from utils import print_separator
import sys
import logging
import pandas as pd
import numpy as np
from time import time
import os

from .utils import *


def get_reduced_df(source_df, feature_df, force_n=None, label="Class"):
    # 整数索引直接当作原始特征名（转成 str）
    idx = feature_df.index
    if idx.dtype in ("int64", "int32"):
        all_feats = [str(i) for i in idx]
        logging.info("feature_df.index 是整数，直接把它们转为字符串列名")
    else:
        all_feats = list(idx.astype(str))
        logging.info("feature_df.index 是字符串，直接使用")

    # 截断前 force_n 个
    sel = all_feats[:force_n] if force_n and force_n > 0 else all_feats

    # 过滤 & 加上标签
    valid = [c for c in sel if c in source_df.columns]
    dropped = set(sel) - set(valid)
    if dropped:
        logging.warning(f"Dropped unknown feats: {dropped}")

    cols = valid + [label]
    reduced = source_df[cols].copy()
    logging.info(f"Reduced to {len(valid)} features + label")

    return reduced


def get_selected_feature_dfs(importances_dir):
    dfs = {}
    for fname in os.listdir(importances_dir):
        if fname.lower().endswith(".csv"):
            full_path = os.path.join(importances_dir, fname)
            if os.path.isfile(full_path):
                dfs[fname] = pd.read_csv(full_path, index_col="feature")
                
    return dfs


def eval_reduced_datasets(run_id, data_path, importances_dir, output_dir, 
                          cv_mode="kfold", k=10, feature_n_config=[-1], clfs=None, random_state=42
    ):
    seed = random_state
    source_df = get_df(data_path)
    feature_dfs = get_selected_feature_dfs(importances_dir)

    out_dir = os.path.join(output_dir, str(seed))
    os.makedirs(out_dir, exist_ok=True)
    
    all_runs = []
    
    for feature_df in feature_dfs:

        logging.info(f"--- seed={seed} ---")
        
        for n in feature_n_config:

            filename = feature_df.split(".csv")[0]
            fs_method = filename.split("_")[0]
            
            print_separator(f"{filename}_{n}_{cv_mode}_{k}")
            
            reduced_df = get_reduced_df(source_df, feature_dfs[feature_df], force_n=n)

            X_e, y_e, feature_names = df_process(reduced_df)
            
            output_file = f"{output_dir}/{seed}/{filename}_scores_{n}-feats_{cv_mode}-cv_{k}-param.csv"

            cvss = evaluate_df(X_e, y_e, output_file=output_file, fs_method=fs_method, run_id=run_id, mode=cv_mode,
                               k=k, clfs=clfs, random_state=seed)

            all_runs.append(cvss)

    result_df = pd.concat(all_runs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_id}_{seed}_baseline_{cv_mode}cv{k}.csv")
    result_df.to_csv(out_path, index=False)
    logging.info(f"Saved all results to {out_path}")

    return result_df