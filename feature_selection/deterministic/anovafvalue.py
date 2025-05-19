# anovafvalue
from sklearn.feature_selection import f_classif
import logging
import numpy as np
def generate_anovafvalue_importances(X_e, y_e, feature_names, dst_path):

    method = "anovafvalue"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    clf = f_classif(X_e, y_e.ravel())
    importances = clf[
        0
    ]  # [0] for statistic (the higher, the better), [1] for p-value (the closest to zero the the better)

    return importances