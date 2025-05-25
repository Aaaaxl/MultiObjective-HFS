import numpy as np
import logging
import pandas as pd
import os
from eval.eval_reduced import get_selected_feature_dfs

from nsgaii.problem import MicroarrayProblem
from nsgaii.crossover import BinaryCrossover
from nsgaii.sampling import FeatureSampling
from nsgaii.mutation import FeatureSamplingMutation


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume

def run_nsgaii_iter(results_data_dir, importances_path,
                    df, y, feature_names,
                    pop_size, n_gen, fs_distrib, n_max,
                    cv_k, prob_multiplier, evaluator_name, evaluator_obj,
                    target_metric,
                    experiment_name,
                    random_state):

    # Load feature importances
    output_path = (f"""{results_data_dir}/{random_state}/nsgaii_solutions/{experiment_name}/""")

    logging.info(f"Run results will be saved in {output_path}.")

    logging.info(f"Loading feature importance sets for {random_state}")

    feature_dfs = get_selected_feature_dfs(importances_dir=importances_path)
    logging.info(f"Files loaded: {[filename for filename in feature_dfs]}")
    feature_dfs = {
        filename.split("_")[0]: feature_dfs[filename] for filename in feature_dfs
    }

    logging.info(f"Creating problem for {random_state}")
    
    # Update reference feature_dfs
    ma_problem = MicroarrayProblem(df, y, cv_k, n_max, evaluator_name, evaluator_obj, target_metric)
    ma_problem.feature_dfs = feature_dfs
    ma_problem.fs_distrib = fs_distrib

    # Define the algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeatureSampling(max_attempts=100, prob_multiplier=prob_multiplier, min_features=2),
        crossover=BinaryCrossover(),
        mutation=FeatureSamplingMutation(),
        eliminate_duplicates=True,
    )

    logging.info(f"Minimizing objectives for {random_state}")
    
    # Optimize the solutions
    res = minimize(ma_problem, algorithm,("n_gen", n_gen),
                   verbose=True, save_history=True, evaluator_name=evaluator_name)

    logging.info(f"Storing results for {random_state}")
    # Prepare historical data
    hist = res.history

    n_evals = []  # corresponding number of function evaluations\
    hist_F = []  # the objective space values in each generation
    hist_cv = []  # constraint violation in each generation
    hist_cv_avg = []  # average constraint violation in the whole population

    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    # Calculate the hypervolume
    approx_ideal = np.min([[indiv.F[0], indiv.F[1]] for indiv in res.pop], axis=0)
    approx_nadir = np.max([[indiv.F[0], indiv.F[1]] for indiv in res.pop], axis=0)

    metric = Hypervolume(
        ref_point=np.array([1.05, 1.05]),
        norm_ref_point=False,
        zero_to_one=True,
        ideal=approx_ideal,
        nadir=approx_nadir,
    )

    hv = [metric.do(_F) for _F in hist_F]

    F = res.F
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    fl = nF.min(axis=0)
    fu = nF.max(axis=0)

    # Prepare to save outputs
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save full population
    full_pop_result = [[indiv.F[0], indiv.F[1]] for indiv in res.pop]

    out_df = pd.DataFrame(full_pop_result, columns=["test_f1_macro", "features"])
    out_df["cv_method"] = "kfold"
    out_df["k_param"] = cv_k
    out_df["model"] = evaluator_name
    # out_df["dataset"] = results_dir.split("/")[-1]
    # out_df["experiment_id"] = experiment_id
    out_df["random_state"] = random_state
    out_df["file"] = ""
    out_df["fs_method"] = f"NSGA-II_{n_gen}gen"
    out_df["fit_time"] = res.exec_time

    out_df.to_csv(f"{output_path}/full_pop.csv")

    # Save full population features
    out_df = pd.DataFrame([indiv.X for indiv in res.pop], columns=feature_names)

    out_df.to_csv(f"""{output_path}/feature_sets.csv""")

    # Save front
    out_df = pd.DataFrame(F, columns=["test_f1_macro", "features"])
    out_df["cv_method"] = "kfold"
    out_df["k_param"] = cv_k
    out_df["model"] = evaluator_name
    # out_df["dataset"] = results_dir.split("/")[-1]
    # out_df["experiment_id"] = experiment_id
    out_df["random_state"] = random_state
    out_df["file"] = ""
    out_df["fs_method"] = f"NSGA-II_{n_gen}gen"
    out_df["fit_time"] = res.exec_time

    out_df.to_csv(f"{output_path}/front.csv")

    # Save all population scores in all generations
    full_pop_hist = [
        [n, i, indiv.F[0], indiv.F[1]]
        for n, algo in enumerate(hist)
        for i, indiv in enumerate(algo.pop)
    ]

    out_df = pd.DataFrame(
        full_pop_hist, columns=["gen", "gen_index", "test_f1_macro", "features"]
    )
    out_df["cv_method"] = "kfold"
    out_df["k_param"] = cv_k
    out_df["model"] = evaluator_name
    # out_df["dataset"] = results_dir.split("/")[-1]
    # out_df["experiment_id"] = experiment_id
    out_df["random_state"] = random_state
    out_df["file"] = ""
    out_df["fs_method"] = f"NSGA-II_{n_gen}gen"
    out_df["fit_time"] = res.exec_time

    out_df.to_csv(f"{output_path}/full_pop_hist.csv")

    # Save hypervolume
    hyperv_array = np.array([hv, n_evals]).T

    out_df = pd.DataFrame(hyperv_array, columns=["hypervolume", "n_evals"])
    out_df.to_csv(f"{output_path}/hypervolume.csv")

    logging.info(f"Stored results of hypervolume.csv, full_pop_hist.csv, front.csv, features_set.csv, full_pop.csv")

    return True
