# mutualinfo
from sklearn.feature_selection import mutual_info_classif
import logging

def generate_mutualinfo_importances(X_e, y_e):

    method = "mutualinfo"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    importances = mutual_info_classif(
        X_e,
        y_e.ravel(),
        n_neighbors=3,
    )

    return importances