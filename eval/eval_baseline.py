from .utils import *

# 对全部特征数据不做特征选择的情况下评估模型
def eval_baseline_df(data_path, output_file, mode='loo', k=5, clfs=None):
    df = get_df(data_path)
    X_e, y_e, feature_names = df_process(df)

    # 交叉验证分数
    cvs = evaluate_df(
        X_e, y_e, output_file=output_file, mode=mode, k=k, clfs=clfs
    )

    return cvs