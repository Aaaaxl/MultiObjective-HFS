from experiment import *
from config import arrhythmia_data_path
from utils import *

def experiment_process(data_path, results_data_dir,
                       min_features, max_features, feature_n_config,
                       cv_mode, cv_k,
                       n_gen, pop_size, fs_prob, fs_distrib,
                       fitness_evaluator_name, fitness_evaluator_obj, fitness_target_metric,
                       clfs, target_metric, random_states):

    experiment_name = f"""nsgaii_{n_gen}-gens_{pop_size}-pop_{cv_k}-k_{min_features}-to-{max_features}-feat_{str(fs_prob).replace(".", "")}-prob_{fitness_evaluator_name}"""
    
    for random_state in random_states:

        importances_path = os.path.join(results_data_dir, 'importances')
    
        output_dir = os.path.join(results_data_dir, 'outputs')
    
        eval_importances_path = os.path.join(importances_path, str(random_state))
        
        generate_baseline_features(data_path, importances_path, random_state)
    
        eval_baseline_features(data_path, eval_importances_path, output_dir, cv_mode, cv_k, feature_n_config, clfs, random_state)
    
        run_moga_optimization(data_path, eval_importances_path, results_data_dir,
                              pop_size, n_gen, fs_distrib, max_features,
                              cv_k, fs_prob, fitness_evaluator_name, fitness_evaluator_obj,
                              fitness_target_metric, experiment_name,
                              random_state)
    
        front_paths, feature_set_paths = get_file_paths(results_data_dir, experiment_name, random_state)
    
        eval_moga_features(data_path, results_data_dir, target_metric, cv_k, clfs, front_paths, feature_set_paths)