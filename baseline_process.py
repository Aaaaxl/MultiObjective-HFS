from feature_selection.feature_selection import feature_selection_all
from eval.eval_reduced import eval_reduced_datasets
from config import *
from utils import *

def baseline_process(run_id, data_path, output_dir, importances_path,
                     cv_mode='kfold', k=5, feature_n_config=[-1], clfs=None, random_states=[42]):
    
    for seed in random_states:
        
        feature_selection_all(data_path, 'Class', importances_path, seed)

        logging.info(f"===={seed} SELECT FINISHED !!! ===")

        eval_importances_path = os.path.join(importances_path, str(seed))
        
        eval_reduced_datasets(run_id, data_path, eval_importances_path, output_dir, 
                              cv_mode, k, feature_n_config, clfs, seed)

        logging.info(f"===={seed} EVAL FINISHED !!! ====")