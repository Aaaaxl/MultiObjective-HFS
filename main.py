from feature_selection.feature_selection import feature_selection_all
from eval.eval_reduced import eval_reduced_datasets
from config import *
from utils import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def baseline_process(args, clf=None):

    feature_n_config = list(range(args.min_features, args.max_features + 1))
    clfs = None
    
    for random_state in args.random_states:

        logging.info(f"=== {random_state} BEGIN TO BASELINE ===")
        
        feature_selection_all(args.data_path, 'Class', args.importances_path, random_state)
            
        logging.info(f"===={random_state} SELECT FINISHED !!! ===")

        output_dir = args.importances_path
        
        eval_importances_path = os.path.join(args.importances_path, str(random_state))
        
        eval_reduced_datasets(args.data_path, eval_importances_path, output_dir, 
                              args.cv_mode, args.cv_k, feature_n_config, clfs, random_state)

        logging.info(f"===={random_state} EVAL FINISHED !!! ====")


from experiment import *
from utils import *

def moga_process(args):
    
    experiment_name = f"""nsgaii_{args.n_gen}-gens_{args.pop_size}-pop_{args.cv_k}-k_{args.min_features}-to-{args.max_features}-feat_{str(args.fs_prob).replace(".", "")}-prob_{args.fitness_evaluator_name}"""

    feature_n_config = list(range(args.min_features, args.max_features + 1))
    
    for random_state in args.random_states:

        fitness_evaluator_obj = LinearSVC(max_iter=10000, dual=True, random_state=random_state)

        clfs = {'LinearSVC': LinearSVC(max_iter=10000, dual=True, random_state=random_state)}
        importances_path = os.path.join(args.results_data_dir, 'importances')
    
        output_dir = os.path.join(args.results_data_dir, 'outputs')
    
        eval_importances_path = os.path.join(args.importances_path, str(random_state))
        
        # generate_baseline_features(args.data_path, args.importances_path, random_state)
    
        # eval_baseline_features(args.data_path, eval_importances_path, output_dir, args.cv_mode, args.cv_k, feature_n_config, clfs, random_state)


        run_moga_optimization(args.data_path, eval_importances_path, args.results_data_dir,
                              args.pop_size, args.n_gen, args.fs_distrib, args.max_features,
                              args.cv_k, args.fs_prob, args.fitness_evaluator_name, fitness_evaluator_obj,
                              args.fitness_target_metric, experiment_name,
                              random_state)


    front_paths, feature_set_paths = get_file_paths(args.results_data_dir, experiment_name, random_state)

    eval_moga_features(args.data_path, args.results_data_dir, args.target_metric, args.cv_k, clfs, front_paths, feature_set_paths)
