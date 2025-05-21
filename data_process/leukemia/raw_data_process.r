# 安装 Bioconductor 和 affy 包
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("affy")  # 用于 Affymetrix CEL 文件

# 加载包
library(affy)

# 读取所有 CEL 文件
data <- ReadAffy(celfile.path = "your_path/Leucemia")

# 背景校正 + 标准化 + 表达量计算
eset <- rma(data)

# 获取表达矩阵
expr_matrix <- exprs(eset)

# 导出为 CSV 文件
write.csv(expr_matrix, file = "expression_matrix_probe.csv")
