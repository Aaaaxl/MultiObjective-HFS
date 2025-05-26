import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.model_selection import StratifiedKFold, cross_validate
from copy import copy
import logging


# 多目标优化问题定义
class MicroarrayProblem(ElementwiseProblem):
    
    def __init__(
        self,
        df,
        y,
        cv_k,
        n_max,
        fitness_evaluator_name,
        fitness_evaluator_obj,
        fitness_target_metric="test_f1_macro",
    ):
        super().__init__(
            n_var=len(df.columns.to_numpy()), n_obj=2, n_ieq_constr=0, xl=0, xu=1
        )
        self.L = df.columns.to_numpy()
        self.n_max = n_max
        self.df = df
        self.X_e = df.to_numpy()
        self.y_e = y
        self.cv_k = cv_k
        self.fitness_evaluator_name = fitness_evaluator_name
        self.fitness_evaluator_obj = fitness_evaluator_obj
        self.fitness_target_metric = fitness_target_metric

    def _evaluate(self, x, out, *args, **kwargs):
        cv = StratifiedKFold(n_splits=self.cv_k, shuffle=True)
        X_e = self.X_e[:, x]
        y_e = self.y_e

        logging.info("BEGIN TO Cross Validate")
        scores = cross_validate(
            copy(self.fitness_evaluator_obj),
            X_e,
            y_e.ravel(),
            cv=cv,
            scoring=[
                # "f1_micro",
                "f1_macro",
                # "f1_weighted",
                # "recall_micro",
                # "recall_macro",
                # "recall_weighted",
                # "accuracy",
            ],
            return_estimator=False,
            n_jobs=-1,
        )

        logging.info("END TO Cross Validate")

        out["F"] = [1 - np.mean(scores[self.fitness_target_metric]), X_e.shape[1]]
