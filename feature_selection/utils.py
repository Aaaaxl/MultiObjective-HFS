import logging
import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

def save_importances(feature_names, importances, method, dst_path, random_state, ascending=False):
    # Save importances to a .csv file following the wTSNE format
    out_dir = os.path.join(dst_path, str(random_state))
    os.makedirs(out_dir, exist_ok=True)
    
    series = pd.Series(feature_names, index=importances).sort_index(ascending=ascending)
    logging.debug(series)

    logging.info(
        f"Saving importances for {method} method with {len(feature_names)} features."
    )
    pd.DataFrame(series, columns=["feature"]).reset_index().rename(
        columns={"index": "value"}
    )[["feature", "value"]].to_csv(
        f"{dst_path}/{random_state}/{method}_{len(feature_names)}_importances.csv", index=False
    )

    return series

# Select features by removing features with no variance
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


def get_df(data_path, index_col=False):
    if index_col == True:
        df = pd.read_csv(data_path, index_col=0)
    else:
        df = pd.read_csv(data_path)

    return df


def df_process(df, label='Class', fit_trans=True):
    # 特征名
    ori_feature_names = df.drop(columns=[label]).columns.tolist()

    # 去掉含 NaN 的列

    # 去掉 X 中的标签列
    X_e = df.drop(columns=[label]).values
    y_e = df[label].values

    if fit_trans == True:
        le = LabelEncoder()
        y_e = le.fit_transform(y_e)  # 变成0/1标签

    
    X_e, mask = select_by_variance(X_e)
    feature_names = [name for name, keep in zip(ori_feature_names, mask) if keep]
    
    return X_e, y_e, feature_names