# mrmr
# from mrmr import mrmr_classif
import mrmr
import logging
import pandas as pd
import numpy as np

def generate_mrmr_importances(X_e, y_e, feature_names, dst_path, k=100):

    method = "mrmr"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")

    selected_features = mrmr.mrmr_classif(
        X=pd.DataFrame(X_e, columns=feature_names),
        y=pd.Series(y_e.ravel()),
        K=k,
        return_scores=True,
    )
    top_k_features = selected_features[0]
    scores = list(range(len(top_k_features), 0, -1))
    print(selected_features)

    importances = np.zeros(len(feature_names))
    
    for i, feat in enumerate(top_k_features):
        try:
            idx = feature_names.index(feat)
            importances[idx] = scores[i]
        except ValueError:
            print(f"[警告] MRMR特征 {feat} 不在 feature_names 中，跳过！")
    
    '''
    feature_pos = np.array(
        [np.where(feature_names == x) for x in top_k_features]
    ).ravel()
    importances[feature_pos] = scores
    '''
    
    return importances