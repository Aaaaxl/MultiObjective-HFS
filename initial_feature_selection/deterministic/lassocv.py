# lassocv
from sklearn.linear_model import LassoCV
import logging
import numpy as np
def generate_lassocv_importances(X_e, y_e, feature_names, dst_path):

    method = "lassocv"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    clf = LassoCV().fit(X_e, y_e.ravel())
    importances = np.abs(clf.coef_)

    return importances