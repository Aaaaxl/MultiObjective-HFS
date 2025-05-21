import os
import sys
from pathlib import Path
from warnings import simplefilter

path_src = str(Path(os.getcwd(), "..").absolute())
sys.path.insert(0, "..")

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=RuntimeWarning)

import numpy as np
from pymoo.core.problem import ElementwiseProblem


# 前 n 个特征置为 True
def get_feature_set_mask(values, n_features):
    # Get first n_features

    indexes = np.full(values.shape, False, dtype=bool)
    indexes[0:n_features] = True
    return indexes


# 根据一组特征的重要性分数，用带随机扰动的方法采样出若干个多样化但又偏向高重要性的特征子集
def get_feature_mask_by_importance(
    values,
    n_samples,
    n_features,
    max_attempts=100,
    prob_multiplier=1.25,
    min_features=None,
):
    # Sort a value for each feature between df.min and df.max (from values).
    # Then select the top n_features that surpass this sorted value.
    # Adds some randomness while favoring top features.

    if min_features:
        size_constraint = min_features
    else:
        size_constraint = n_features

    if n_features >= values.shape[0]:
        raise Exception("Not enough features (n_features >= len(values))")

    selected_indexes = []
    selected_index_masks = []
    for sample_n in range(n_samples):
        selected_index = []
        attempts = 0

        # While selected_index is empty or another identical selected_index was already selected
        while len(selected_index) == 0 or np.any(
            [np.all(selected_index == item) for item in selected_indexes]
        ):
            selected_index = []
            while len(selected_index) < size_constraint:
                values_ = values.copy()
                # Set values for features already selected
                values_[selected_index] = np.min(values_)
                # Randomize values to select new candidate features (not selected yet)
                chances = np.random.uniform(
                    low=np.min(values_),
                    high=np.max(values_) * prob_multiplier,
                    size=values_.shape,
                )
                # Set chances for features already selected
                chances[selected_index] = np.min(values_)
                # Verify chosen features
                results = [
                    True if x >= y else False for (x, y) in zip(values_, chances)
                ]
                # Select top n_features
                selected_index = np.argwhere(results)[0:n_features, 0]

                # if min_features is set, select random final_n_features between min_features and n_features
                if min_features and len(selected_index) >= size_constraint:
                    upper_limit = min(n_features, len(selected_index))
                    final_n_features = (
                        np.random.randint(min_features, upper_limit)
                        if upper_limit > min_features
                        else min_features
                    )
                    selected_index = sorted(
                        np.random.choice(
                            selected_index, final_n_features, replace=False
                        )
                    )

                # if the feature set already exists in the collection, try flipping a bit to force diversion
                while np.any(
                    [np.all(selected_index == item) for item in selected_indexes]
                ):
                    # if min_features is set
                    if min_features and len(selected_index) >= size_constraint:
                        upper_limit = min(n_features, len(selected_index))
                        final_n_features = (
                            np.random.randint(min_features, upper_limit)
                            if upper_limit > min_features
                            else min_features
                        )
                        selected_index = sorted(
                            np.random.choice(
                                selected_index, final_n_features, replace=False
                            )
                        )
                    else:
                        final_n_features = n_features

                    # if the selected_index is present in the collection, try flipping a bit to force diversion
                    if (
                        np.any(
                            [
                                np.all(selected_index == item)
                                for item in selected_indexes
                            ]
                        )
                        and len(selected_index) >= size_constraint
                    ):
                        flip_position = np.random.choice(selected_index)
                        results[flip_position] = not results[flip_position]
                        selected_index = np.argwhere(results)[0:final_n_features, 0]

                    # Limit attempts to a maximum amount. Infinite loops are not yet defined.
                    attempts += 1
                    if attempts >= max_attempts:
                        raise Exception(
                            f"Attempts to generate new feature sets reached max_attempts ({max_attempts}). Amount of sets created: {len(selected_index_masks)}"
                        )

        top_n_selected_index = selected_index
        selected_indexes.append(top_n_selected_index)
        index_mask = np.full(values.shape[0], False, dtype=bool)
        index_mask[top_n_selected_index] = True
        selected_index_masks.append(index_mask)

    return np.array(selected_index_masks)


# 排名采样
def get_feature_mask_by_rank(values, n_samples, max_features, min_features=1):
    """
    Generates n_samples features masks starting with top min_features and, for each next sample until n_samples are created, adding the next subsequent feature.
    """

    print(f"Generating {n_samples} feature sets.")

    if n_samples > len(range(min_features, max_features + 1)):
        raise Exception(
            "Not enough features to complete the sample set (n_samples > diff(min_features, max_features))"
        )

    selected_index_masks = []
    for sample_n in range(min_features, n_samples + 1):
        index_mask = np.full(values.shape[0], False, dtype=bool)
        top_n_selected_index = range(0, sample_n)
        index_mask[top_n_selected_index] = True
        selected_index_masks.append(index_mask)

    print(
        f"Generated feature sets with distribution: {[np.sum(mask) for mask in selected_index_masks]}"
    )

    return np.array(selected_index_masks)