# decisiontree
from sklearn.tree import DecisionTreeClassifier
import logging

def generate_decisiontree_importances(X_e, y_e, random_state):

    method = "decisiontree"
    logging.info(f"Running {method} method for dataset with shape {X_e.shape}, random_state={random_state}.")

    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_e, y_e.ravel())

    importances = clf.feature_importances_
    return importances