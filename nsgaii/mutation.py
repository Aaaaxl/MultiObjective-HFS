import numpy as np
from pymoo.core.mutation import Mutation
from .utils import get_feature_mask_by_importance

# 扰动
class NoMutation(Mutation):
    
    def __do(self, problem, X, **kwargs):
        return X

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False
        print(f"Mutated {len(X)} individuals.")

        return X


# 变异
class FeatureSamplingMutation(Mutation):
    
    def __do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False
        print(f"Mutated {len(X)} individuals.")

        return X

    def _do(self, problem, X, **kwargs):

        feature_dfs = {...}
        fs_distrib = {"mrmr": 0.2, "relieff": 0.2, "kruskalwallis": 0.2, "mutualinfo": 0.2, "decisiontree": 0.2}
        """
        # 采样特征选择分数
        feature_dfs = problem.feature_dfs
        fs_distrib = problem.fs_distrib
        methods = [method for method in fs_distrib]

        feature_df = feature_dfs[np.random.choice(methods)]

        # 采样“优秀特征mask”
        feature_mask = get_feature_mask_by_importance(
            values=feature_df.values,
            n_samples=1,
            n_features=problem.n_max,
            max_attempts=100,
            prob_multiplier=1,
            min_features=problem.n_max,
        )

        # 排序对齐
        idx1 = np.argsort(problem.df.columns)
        idx2 = np.argsort(feature_df.index)
        idx1_inv = np.argsort(idx1)
        feature_mask = feature_mask[:, idx2][:, idx1_inv]

        mask_true = np.where(feature_mask)[1]

        # 对每个个体进行变异
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_true = np.where(X[i, :])[0]
            is_false = np.where(np.logical_not(X[i, :]))[0]

            available_features = np.intersect1d(mask_true, is_false)

            if len(available_features) > 0:
                # Enables one feature
                X[i, np.random.choice(available_features)] = True
                # Disables one feature
                X[i, np.random.choice(is_true)] = False
        print(f"Mutated {len(X)} individuals with feature sampling.")

        return X
