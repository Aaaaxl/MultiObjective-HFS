from utils import print_separator
import sys
import logging
import pandas as pd
import numpy as np
from time import time
import os

from .utils import *


def get_reduced_df(source_df, feature_df, force_n=None, label="Class"):

    idx = feature_df.index
    if idx.dtype in ("int64", "int32"):
        all_feats = [str(i) for i in idx]
        logging.info("feature_df.index 是整数，直接把它们转为字符串列名")
    else:
        all_feats = list(idx.astype(str))
        logging.info("feature_df.index 是字符串，直接使用")

        # 尝试将 force_n 转为整数，如果失败则使用所有特征
    try:
        n = int(force_n)
    except (TypeError, ValueError):
        logging.warning(f"force_n 不是合法整数，使用所有特征（force_n={force_n}）")
        n = None
        
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
    """
    根据文件名前缀（FS 方法名）建立字典，key 为方法名，如 'mrmr', 'relieff' 等
    """
    dfs = {}
    for fname in os.listdir(importances_dir):
        if not fname.lower().endswith('.csv'):
            continue
        method = fname.split('_')[0]  # 提取方法名
        full_path = os.path.join(importances_dir, fname)
        if os.path.isfile(full_path):
            dfs[method] = pd.read_csv(full_path, index_col='feature')
    logging.info(f"Loaded feature_dfs for methods: {list(dfs.keys())}")
    return dfs



def eval_reduced_datasets(data_path, importances_dir, output_dir,
                          cv_mode="kfold", k=10, feature_n_config=None, clfs=None, random_state=42):
    if feature_n_config is None:
        feature_n_config = list(range(2, 51))
    seed = random_state
    source_df = get_df(data_path)
    feature_dfs = get_selected_feature_dfs(importances_dir)

    out_dir = os.path.join(output_dir, str(seed))
    os.makedirs(out_dir, exist_ok=True)
    all_runs = []

    for method, feature_df in feature_dfs.items():
        logging.info(f"--- seed={seed} | method={method} ---")
        for n in feature_n_config:
            assert isinstance(n, int), f"n 不是整数: {n}"
            print_separator(f"{method}_{n}_{cv_mode}_{k}")

            reduced_df = get_reduced_df(source_df, feature_df, force_n=n)
            X_e, y_e, feature_names = df_process(reduced_df)

            # 标签编码
            le = LabelEncoder()
            y_enc = le.fit_transform(y_e.ravel())

            output_file = f"{out_dir}/{method}_scores_{n}-feats_{cv_mode}-cv_{k}.csv"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                cvss = evaluate_df(
                    X_e, y_enc,
                    output_file=output_file,
                    fs_method=method,
                    mode=cv_mode,
                    k=k,
                    clfs=clfs,
                    random_state=seed
                )
            all_runs.append(cvss)

    result_df = pd.concat(all_runs, ignore_index=True)
    out_path = os.path.join(output_dir, f"{seed}_baseline_{cv_mode}cv{k}.csv")
    result_df.to_csv(out_path, index=False)
    logging.info(f"Saved all results to {out_path}")
    return result_df
