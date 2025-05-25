import numpy as np
import pandas as pd
import os
import logging

from sklearn.model_selection import StratifiedKFold, cross_validate

def eval_fronts_feature_sets(front_df, feature_list_df, X, y, clf=None, cv_k=5):
    evaluations = []

    for front_id in front_df.index:
        feature_list = (
            feature_list_df.loc[front_id]
            .where(feature_list_df.loc[front_id] == True)
            .dropna()
            .index
        )

        X_selected = X[feature_list].copy()
        y = y

        scores = evaluate(X_selected, y, clf, cv_k)
        evaluations.append({"front_id": front_id, "scores": scores})

    return evaluations


def get_target_from_front_evals(evals, target):
    metric = [(item["front_id"], np.mean(item["scores"][target])) for item in evals]

    return metric


def evaluate(X_e, y_e, clf, cv_k):
    cv = StratifiedKFold(n_splits=cv_k, shuffle=True)

    scores = cross_validate(
        clf,
        X_e,
        y_e,
        cv=cv,
        scoring=[
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "accuracy",
        ],
        return_estimator=False,
        n_jobs=-1,
    )

    return scores


'''
def get_file_paths(results_data_dir, experiment_name, random_state):

    subdir_iterators = [
        os.walk(
            f"{results_data_dir}/{random_state}/nsgaii_solutions/{experiment_name}"
        )
    ]
    subdir_files = [
        os.path.join(dirpath, filename)
        for subdir_iterator in subdir_iterators
        for (dirpath, dirnames, filenames) in subdir_iterator
        for filename in filenames
    ]

    expected_dir_name = "front"
    logging.info(f"Looking for files with prefix: {expected_dir_name}.")

    front_paths = [
        subdir_file
        for subdir_file in subdir_files
        if (expected_dir_name in subdir_file)
    ]
    logging.info(f"Recovering all front files: {front_paths}.")

    expected_dir_name = "feature_sets"
    logging.info(f"Looking for files with suffix: {expected_dir_name}.")

    feature_set_paths = [
        subdir_file
        for subdir_file in subdir_files
        if (expected_dir_name in subdir_file)
    ]
    logging.info(f"Recovering all front files: {feature_set_paths}.")

    return front_paths, feature_set_paths
'''

def get_file_paths(results_data_dir, experiment_name, random_state):
    # 构造 base_path
    base_path = os.path.join(
        results_data_dir,
        str(random_state),
        "nsgaii_solutions",
        experiment_name
    )
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"目录不存在: {base_path}")

    # 遍历所有文件
    subdir_files = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(base_path)
        for filename in filenames
        if filename.lower().endswith(".csv")
    ]

    # 只匹配文件名以 "front" 开头的
    logging.info("Looking for files with prefix: front.")
    front_paths = [
        p for p in subdir_files
        if os.path.basename(p).startswith("front")
    ]
    logging.info(f"Recovering all front files: {front_paths}.")

    # 只匹配文件名以 "feature_sets" 开头的
    logging.info("Looking for files with prefix: feature_sets.")
    feature_set_paths = [
        p for p in subdir_files
        if os.path.basename(p).startswith("feature_sets")
    ]
    logging.info(f"Recovering all feature_set files: {feature_set_paths}.")

    return front_paths, feature_set_paths


