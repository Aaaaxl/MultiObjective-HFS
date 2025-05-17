# decisiontree
from sklearn.tree import DecisionTreeClassifier
import logging

def generate_decisiontree_importances(X_e, y_e):

    method = "decisiontree"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    clf = DecisionTreeClassifier().fit(X_e, y_e.ravel())
    importances = clf.feature_importances_

    return importances