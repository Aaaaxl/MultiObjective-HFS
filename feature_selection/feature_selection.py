import pandas as pd

from .non_deterministic import *
from .deterministic import *
from .utils import *


def feature_selection(X_e, y_e, method, feature_names, dst_path, random_state, ascending=False):
    # non_deterministic
    if method == 'decisiontree':
        importances = generate_decisiontree_importances(X_e, y_e, random_state)

    elif method == 'randomforest':
        importances = generate_randomforest_importances(X_e, y_e, random_state)

    elif method == 'linearsvm':
        importances = generate_linearsvm_importances(X_e, y_e, random_state)

    elif method == 'mutualinfo':
        importances = generate_mutualinfo_importances(X_e, y_e, random_state)

    # deterministic
    elif method == 'anovafvalue':
        importances = generate_anovafvalue_importances(X_e, y_e, feature_names, dst_path)
        
    elif method == 'kruskalwallis':
        importances = generate_kruskalwallis_importances(X_e, y_e, feature_names, dst_path)
        
    elif method == 'lassocv':
        importances = generate_lassocv_importances(X_e, y_e, feature_names, dst_path)
        
    elif method == 'mrmr':
        importances = generate_mrmr_importances(X_e, y_e, feature_names, dst_path)

    elif method == 'relieff':
        importances = generate_relieff_importances(X_e, y_e, feature_names, dst_path)

    # others
    else:
        raise ValueError(f"你这 {method} 方法还没实现呢，想用快自己代码！")

    sorted_importances = save_importances(feature_names, importances, method, dst_path, random_state, ascending=ascending)

    return sorted_importances


def feature_selection_all(data_path, label, importances_path,
                          random_state=42, fit_trans=False, index_col=False):
    # method in literature
    deterministic = ['kruskalwallis', 
                     'mrmr',
                     'lassocv', 
                     'anovafvalue', 
                     'relieff']
    non_deterministic = ['decisiontree', 
                         'randomforest', 
                         'linearsvm', 
                         'mutualinfo']

    df = get_df(data_path)
    X_e, y_e, ori_feature_names = df_process(df)

    X_e, mask = select_by_variance(X_e)
    feature_names = [name for name, keep in zip(ori_feature_names, mask) if keep]
    
    for method in non_deterministic:
        sorted_importances = feature_selection(X_e, y_e, method, 
                                               feature_names, importances_path, random_state)
    logging.info(f"{data_path}_{random_state}_Non-deterministic 全部搞定")
    
    for method in deterministic:
        sorted_importances = feature_selection(X_e, y_e, method, 
                                               feature_names, importances_path, random_state)
    logging.info(f"{data_path}_{random_state}_Deterministic 全部搞定")