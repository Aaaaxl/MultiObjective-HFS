import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# Breast Cancer Data Path
breast_raw_data_path = r'C:\Users\26494\PycharmProjects\MOO-HFS\dataset\Breast_Cancer'

breast_expression_probe = os.path.join(DATA_DIR, "Breast_Cancer/expression_matrix_probe.csv")

breast_expression_gene = os.path.join(DATA_DIR, "Breast_Cancer/expression_matrix_gene.csv")

breast_data_path = os.path.join(DATA_DIR, "Breast_Cancer/Breast_Cancer.csv")

breast_importances_path = os.path.join(DATA_DIR, "importances/Breast_Cancer")

# Leukemia Data Path
leukemia_raw_data_path = r'C:\Users\26494\PycharmProjects\MOO-HFS\dataset\Leukemia'

leukemia_expression_path = os.path.join(DATA_DIR, 'Leukemia\expression_matrix.csv')
                                        
leukemia_annotation_path = os.path.join(DATA_DIR, 'Leukemia\GSE28497_sample_annotation.txt.gz')
                                        
leukemia_series_matrix_path = os.path.join(DATA_DIR, 'Leukemia\GSE28497_series_matrix.txt.gz')

leukemia_data_path = os.path.join(DATA_DIR, 'Leukemia\leukemia.csv')
                                           
leukemia_importances_path = os.path.join(DATA_DIR, "importances/Leukemia")

# p53_Mutants
p53_raw_data_path = r'C:\Users\26494\PycharmProjects\MOO-HFS\dataset\p53_Mutants\p53_old_2010'

p53_data_path = os.path.join(DATA_DIR, 'p53_Mutants/p53_Mutants.csv')

p53_importances_path = os.path.join(DATA_DIR, 'importances/p53_Mutants')


# Arrhythmia Data Path
arrhythmia_raw_data_path = r'C:\Users\26494\PycharmProjects\MOO-HFS\dataset\Arrhythmia'

arrhythmia_data_path = os.path.join(DATA_DIR, 'arrhythmia/arrhythmia.csv')

arrhythmia_importances_path = os.path.join(DATA_DIR, 'importances/arrhythmia')

arrhythmia_results_path = os.path.join(RESULT_DIR, 'arrhythmia')