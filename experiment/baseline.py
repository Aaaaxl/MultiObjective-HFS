import logging
import os
import shutil
import pandas as pd
from eval.eval_reduced import eval_reduced_datasets, get_selected_feature_dfs
from feature_selection.feature_selection import feature_selection_all

def generate_baseline_features(data_path, importances_path, random_state, duplicate_methods=None):

    feature_selection_all(data_path, 'Class', importances_path, random_state=random_state)

    # 写入不用重复运行的方法，避免冗余
    if duplicate_methods is not None:
        raise (NotImplementedError)


def eval_baseline_features(data_path, importances_path, output_dir, cv_mode, cv_k, feature_n_config, clfs=None, random_state=42):

    # Run evaluation
    eval_reduced_datasets(data_path=data_path, importances_dir=importances_path, output_dir=output_dir,
                          cv_mode=cv_mode, k=cv_k, feature_n_config=feature_n_config, clfs=clfs,
                          random_state=random_state)