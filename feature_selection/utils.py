import logging
import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder


def save_importances(feature_names, importances, method, dst_path, random_state, ascending=False):
    # Save importances to a .csv file following the wTSNE format
    out_dir = os.path.join(dst_path, str(random_state))
    os.makedirs(out_dir, exist_ok=True)
    
    series = pd.Series(feature_names, index=importances).sort_index(ascending=ascending)
    logging.debug(series)

    logging.info(
        f"Saving importances for {method} method with {len(feature_names)} features."
    )
    pd.DataFrame(series, columns=["feature"]).reset_index().rename(
        columns={"index": "value"}
    )[["feature", "value"]].to_csv(
        f"{dst_path}/{random_state}/{method}_{len(feature_names)}_importances.csv", index=False
    )

    return series