# linearsvm
from sklearn.svm import LinearSVC
import logging
import numpy as np

def generate_linearsvm_importances(X_e, y_e, gridsearch=False):

    method = "linearsvm"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    if gridsearch:
        logging.info("Using GridSearch.")
        param_grid = {
            "tol": [1e-6, 1e-4, 1e-2],
            "C": [1, 10],
            "max_iter": [1000, 2000],
            "dual": [True],
        }
        grid_cv = GridSearchCV(LinearSVC(), param_grid).fit(X_e, y_e.ravel())
        logging.info(f"Params chosen by GridSearch: {grid_cv.best_params_}")

        clf = grid_cv.best_estimator_
    else:
        clf = LinearSVC(dual=True).fit(X_e, y_e.ravel())

    svm_weights = np.abs(clf.coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()
    importances = svm_weights

    return importances