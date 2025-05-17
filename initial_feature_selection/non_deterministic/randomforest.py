# randomforest
from sklearn.ensemble import RandomForestClassifier
import logging

def generate_randomforest_importances(X_e, y_e, gridsearch=False):

    method = "randomforest"

    logging.info(f"Running {method} method for dataset with shape {X_e.shape}.")
    if gridsearch:
        logging.info("Using GridSearch.")
        param_grid = {
            "n_estimators": [100, 150],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 3],
        }
        grid_cv = GridSearchCV(RandomForestClassifier(), param_grid).fit(
            X_e, y_e.ravel()
        )
        logging.info(f"Params chosen by GridSearch: {grid_cv.best_params_}")

        clf = grid_cv.best_estimator_
    else:
        # Using a larger number of estimators by standard (default is 100)
        clf = RandomForestClassifier(
            n_estimators=150,
        ).fit(X_e, y_e.ravel())

    importances = clf.feature_importances_

    return importances