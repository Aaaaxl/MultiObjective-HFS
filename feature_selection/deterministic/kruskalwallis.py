# kruskalwallis
from scipy.stats import kruskal
import logging
import numpy as np

def generate_kruskalwallis_importances(X_e, y_e, feature_names, dst_path):

    method = "kruskalwallis"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    label_groups = (X_e[np.argwhere(y_e == label)[:, 0]] for label in np.unique(y_e))
    res = kruskal(*label_groups)
    importances = (
        res.statistic
    )  # statistic, pvaluer for p-value (the closest to zero the the better)

    return importances