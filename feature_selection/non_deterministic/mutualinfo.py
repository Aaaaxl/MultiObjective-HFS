# mutualinfo
from sklearn.feature_selection import mutual_info_classif
import logging

def generate_mutualinfo_importances(X_e, y_e, random_state, n_neighbors=3):

    method = "mutualinfo"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}, n_neighbors={n_neighbors}, random_state={random_state}.")
    
    importances = mutual_info_classif(
        X_e,
        y_e.ravel(),
        n_neighbors=n_neighbors,
        random_state=random_state
    )

    return importances