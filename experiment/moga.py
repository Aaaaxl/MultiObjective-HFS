import logging
import os
import numpy as np
import pandas as pd

from utils import *
from feature_selection.utils import get_df, df_process
from eval.eval_reduced import get_selected_feature_dfs
from .nsgaii_iter import run_nsgaii_iter
from .utils import eval_fronts_feature_sets
from .utils import get_target_from_front_evals



def run_moga_optimization(data_path, importances_path, results_data_dir,
                          pop_size, n_gen, fs_distrib, n_max,
                          cv_k, prob_multiplier, evaluator_name, evaluator_obj,
                          target_metric,
                          experiment_name, random_state):
    # 数据读取和预处理
    df = get_df(data_path)
    X_e, y_e, feature_names = df_process(df)
    data = pd.DataFrame(X_e, columns=feature_names, index=df.index)

    logging.info(f"Running MOGA for dataset with shape {data.shape} for random_state {random_state}")

    run_nsgaii_iter(results_data_dir, importances_path, data, y_e, feature_names,
                    pop_size, n_gen, fs_distrib, n_max,
                    cv_k, prob_multiplier, evaluator_name, evaluator_obj,
                    target_metric, experiment_name,  random_state)


# 对多目标特征选择算法产生的“帕累托前沿”特征子集，做交叉验证评估
def eval_moga_features(data_path, results_data_dir, target_metric, cv_k, clfs, front_paths, feature_set_paths):
    df = get_df(data_path)

    # 读取每个“帕累托前沿”结果和特征子集
    front_dfs = [pd.read_csv(front_path, index_col=0) for front_path in front_paths]
    feature_set_dfs = [
        pd.read_csv(feature_set_path, index_col=0)
        for feature_set_path in feature_set_paths
    ]
    new_front_dfs = []

    # 数据预处理
    X_e, y_e, feature_names = df_process(df)
    df = pd.DataFrame(X_e, columns=feature_names, index=df.index)

    # 评估
    for clf_name, clf in clfs.items():
        for front_df, feature_list_df in zip(front_dfs, feature_set_dfs):
            evals = eval_fronts_feature_sets(
                front_df, feature_list_df, df, y_e, clf, cv_k
            )
            scores = get_target_from_front_evals(evals, target_metric)

            target_metrics_df = pd.DataFrame(
                scores, columns=["front_id", target_metric]
            )
            target_metrics_df["model"] = clf_name

            front_df[target_metric] = target_metrics_df[target_metric]
            front_df["model"] = clf_name
            new_front_dfs.append(front_df)

    # 汇总结果
    front_evals_df = pd.concat(new_front_dfs)

    # 保存结果
    if not os.path.exists(f"""{results_data_dir}/nsgaii_results/"""):
        os.makedirs(f"""{results_data_dir}/nsgaii_results/""")

    front_evals_df.to_csv(f"""{results_data_dir}/nsgaii_results/cv_results_assembled.csv""")