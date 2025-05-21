import sys
import logging
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold


# 简单的输出分隔符
def print_separator(text, sep="=", total_len=50):
    text = f" {text} "
    side_len = (total_len - len(text)) // 2
    print('\n' + sep * side_len + text + sep * (total_len - side_len - len(text)))


# 设置 logging
def set_logger():
    # 拿到根 logger 并清空已有 handler
    logger = logging.getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    # 新建一个输出到 stdout 的 handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # 格式化
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    
    # 挂回根 logger，并设置级别
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# 去除方差为 0 的特征列
def select_by_variance(X_e):
    selector = VarianceThreshold()
    selector_f = selector.fit(X_e)
    X_e_transformed = selector_f.transform(X_e)
    X_e_transformed_feature_names = selector_f.get_support()

    logging.info(f"Original data shape: {X_e.shape}")
    logging.info(f"Selected data shape: {X_e_transformed.shape}")
    diff = X_e.shape[1] - X_e_transformed.shape[1]
    logging.info(
        f"Diff: {diff} features, Excluded proportion: {diff/X_e.shape[1]}, Resulting proportion: {1 - diff/X_e.shape[1]}"
    )

    return X_e_transformed, X_e_transformed_feature_names


# 得到 dataframe 格式的数据
def get_df(data_path, index_col=False):
    if index_col == True:
        df = pd.read_csv(data_path, index_col=0)
    else:
        df = pd.read_csv(data_path)

    return df


# 从 df 中提取 X_e, y_e, feature_names
def df_process(df, label='Class', fit_trans=True):
    # 特征名
    ori_feature_names = df.drop(columns=[label]).columns.tolist()

    # 去掉 X 中的标签列
    X_e = df.drop(columns=[label]).values
    y_e = df[label].values

    if fit_trans == True:
        le = LabelEncoder()
        y_e = le.fit_transform(y_e)  # 变成0/1标签

    
    X_e, mask = select_by_variance(X_e)
    feature_names = [name for name, keep in zip(ori_feature_names, mask) if keep]
    
    return X_e, y_e, feature_names