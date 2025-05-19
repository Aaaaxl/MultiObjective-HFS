import sys
import logging
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder


def print_separator(text, sep="=", total_len=50):
    text = f" {text} "
    side_len = (total_len - len(text)) // 2
    print('\n' + sep * side_len + text + sep * (total_len - side_len - len(text)))


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