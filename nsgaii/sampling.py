import numpy as np
import math
from pymoo.core.sampling import Sampling
from .utils import get_feature_mask_by_importance
from .utils import get_feature_mask_by_rank


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[: problem.n_max]
            X[k, I] = True

        return X


# 根据多种特征选择方法的分数，生成多样化的“高质量”初始种群
class FeatureSampling(Sampling):
    def __init__(self, max_attempts=100, prob_multiplier=1.25, min_features=2):
        self.max_attempts = max_attempts
        self.prob_multiplier = prob_multiplier
        self.min_features = min_features
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):

        feature_dfs = problem.feature_dfs
        fs_distrib = problem.fs_distrib
        methods = [method for method in fs_distrib]
        
        print("可用的方法：", list(feature_dfs.keys()))
        
        X = []
        for method in methods:
            feature_df, fs_proportion = feature_dfs[method], fs_distrib[method]

            X_ = get_feature_mask_by_importance(
                values=feature_df.values,
                n_samples=int(n_samples * fs_proportion),
                n_features=problem.n_max,
                max_attempts=self.max_attempts,
                prob_multiplier=self.prob_multiplier,
                min_features=self.min_features,
            )

            # Reorder the resulting array - the feature_df does not use the same sorting as the original problem.df
            idx1 = np.argsort(problem.df.columns)
            idx2 = np.argsort(feature_df.index)
            idx1_inv = np.argsort(idx1)
            X_ = X_[:, idx2][:, idx1_inv]

            X.append(X_)

        X = np.concatenate(X)
        print(f"Generated {len(X)} new samples.")

        return X


# 混入多种方法的随机采样
class StrictFeatureSampling(Sampling):
    def __init__(self, best_fs_set, min_features=2):
        self.best_fs_set = best_fs_set
        self.min_features = min_features
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):

        feature_dfs = problem.feature_dfs
        fs_distrib = problem.fs_distrib
        methods = [method for method in fs_distrib]

        # 用最优方法生成“前半部分”确定性子集
        X = []
        n_samples_ = min(n_samples, problem.n_max - self.min_features)
        feature_df = feature_dfs[self.best_fs_set]
        X_ = get_feature_mask_by_rank(
            values=feature_df.values,
            n_samples=n_samples_,
            max_features=problem.n_max,
            min_features=self.min_features,
        )

        # 顺序对齐到主数据集
        idx1 = np.argsort(problem.df.columns)
        idx2 = np.argsort(feature_df.index)
        idx1_inv = np.argsort(idx1)
        X_ = X_[:, idx2][:, idx1_inv]
        X.append(X_)

        # 剩余子集用随机重要性采样，保证多样性
        n_samples_remaining = max(n_samples - n_samples_, 0)
        X_ = []
        if n_samples_remaining > 0:
            for method in methods:
                feature_df, fs_proportion = feature_dfs[method], fs_distrib[method]

                X__ = get_feature_mask_by_importance(
                    values=feature_df.values,
                    n_samples=math.ceil(n_samples_remaining * fs_proportion),
                    n_features=problem.n_max,
                    max_attempts=self.max_attempts,
                    prob_multiplier=self.prob_multiplier,
                    min_features=self.min_features,
                )

                # 顺序对齐到主数据集
                idx1 = np.argsort(problem.df.columns)
                idx2 = np.argsort(feature_df.index)
                idx1_inv = np.argsort(idx1)
                X__ = X__[:, idx2][:, idx1_inv]
                X_.append(X__)

        X.append(X_[0:n_samples_remaining])
        X = np.concatenate(X)

        print(f"Generated {len(X)} new samples.")

        return X