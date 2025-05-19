# relieff
from skrebate import ReliefF
import logging

def generate_relieff_importances(X_e, y_e, feature_names, dst_path):

    method = "relieff"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    reliefF_clf = ReliefF(n_features_to_select=15, n_neighbors=100, n_jobs=-1).fit(
        X_e, y_e.ravel()
    )
    importances = reliefF_clf.feature_importances_

    return importances