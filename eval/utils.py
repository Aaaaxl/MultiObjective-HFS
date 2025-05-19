import sys
import logging
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, LeavePOut, StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from time import time
import os


def get_df(data_path, index_col=False):
    if index_col == True:
        df = pd.read_csv(data_path, index_col=0)
    else:
        df = pd.read_csv(data_path)

    return df


def df_process(df, label='Class', fit_trans=False):
    # 特征名
    feature_names = df.drop(columns=[label]).columns.tolist()

    # 去掉含 NaN 的列

    # 去掉 X 中的标签列
    X_e = df.drop(columns=[label]).values
    y_e = df[label].values

    if fit_trans == True:
        le = LabelEncoder()
        y_e = le.fit_transform(y_e)  # 变成0/1标签

    return X_e, y_e, feature_names


# 分类器
def get_eval_model(random_state):
    clfs = {}
    clfs["linearsvm"] = LinearSVC(max_iter=3000, dual=True, random_state=random_state)

    return clfs


def evaluate_df(X_e, y_e, output_file=None, filename=None, fs_method=None, run_id=None, mode="loo", k=5, clfs=None, random_state=42):                
    if clfs is None:
        clfs = get_eval_model(random_state)

    # 选择交叉验证策略
    if mode == "loo":
        cv = LeaveOneOut()
    elif mode == "lpo":
        cv = LeavePOut(p=k)
    elif mode == "kfold":
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    # 遍历每个分类器
    cv_scores = []
    for name, clf in clfs.items():
        logging.info(f"Evaluating {name} using {mode} strategy (seed={random_state}).")

        tic_fwd = time()
        scores = cross_validate(clf, X_e, y_e.ravel(), cv=cv,
                                scoring=[
                                    # "f1_micro", 
                                    "f1_macro", 
                                    # "f1_weighted",
                                    # "precision_micro", "precision_macro", "precision_weighted",
                                    # "recall_micro", 
                                    # "recall_macro", 
                                    # "recall_weighted",
                                    # "accuracy",
                                    ],
                                return_estimator=True,
                                n_jobs=-1)
        toc_fwd = time()
        logging.info(f"Evaluated {name} in {toc_fwd - tic_fwd:.3f}s")

        logging.debug(f"Scores for {name}: {scores}")
        scores["cv_method"] = mode
        scores["k_param"] = k
        scores["features"] = X_e.shape[1]
        scores["model"] = name
        # scores["dataset"] = results_dir.split("/")[-1]
        # scores["experiment_id"] = experiment_id
        scores["run_id"] = run_id
        # scores["file"] = filename
        # scores["fs_method"] = fs_method
        scores["random_state"] = random_state
        cv_scores.append(pd.DataFrame.from_dict(scores))

    cv_scores_df = pd.concat(cv_scores)

    metrics = "test_f1_macro"
    mean_value = cv_scores_df[metrics].mean()

    logging.info(f"f1_macro_mean: {mean_value}")
    logging.debug(f"Final CV scores: {cv_scores}")
    logging.info(f"Saving scores in: {output_file}")
    cv_scores_df.to_csv(output_file)

    return cv_scores_df