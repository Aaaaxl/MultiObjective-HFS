# 🌟 MultiObjective-HFS

本仓库收录了《计算智能》课程第7组阅读文献复现并改进相关算法的全部代码。

> **原文献**：[《Enhancing classification with hybrid feature selection: A multi-objective genetic algorithm for high-dimensional data》](https://www.sciencedirect.com/science/article/pii/S095741742401385X)  
> **作者源码**：[sbcblab/MOO-HFS](https://github.com/sbcblab/MOO-HFS)

---

## ✅ 当前实现的功能

- [x] Arrhythmia、p53_Mutants、Breast Cancer 数据集预处理
- [x] 传统方法特征选择、分类（baseline 实验）
- [x] 多目标遗传算法 (MOGA 实验) 的主要模块实现

---

## 🛠️ 已进行的改进

- [x] 优化代码结构，特别是添加随机种子，结果可复现
- [x] 增加详细注释与说明文档
- [ ] 提升方法的性能
- [ ] 补充...

---

## 📁 目录结构

```
MOO-HFS/
│
├── data_process/            # 处理数据代码部分
│   ├── arrhythmia/
│   │   ├── raw_data_process.ipynb 
│   │   └── arrhythmia_des.md
│   │
│   ├── breast_cancer/
│   │   ├── raw_data_preprocess.ipynb
│   │   ├── merge_data_label.ipynb
│   │   ├── data_check.ipynb
│   │   └── breast_des.md
│   │   │ 
│   ├── p53_mutants/
│   │   ├── raw_data_process.ipynb
│   │   └── p53_des.md
│   │ 
│   └── readme.md
│
├── feature_selection/            # 特征选择部分
│   │ 
│   ├── deterministic/          # 确定性特征选择方法
│   │   ├── anovafvalue.py
│   │   ├── kruskalwallis.py
│   │   ├── lassocv.py
│   │   ├── mrmr.py
│   │   └── relieff.py
│   │
│   ├── non_deterministic/          # 非确定性特征选择方法
│   │   ├── decisiontree.py
│   │   ├── linearsvm.py
│   │   ├── mutualinfo.py
│   │   └── randomforest.py
│   │
│   ├── feature_selection.py            # 特征选择主函数
│   └── utils.py            # 辅助函数
│
├── eval/            # 验证方法目录
│   ├── eval_baseline.py            # 不使用特征选择方法
│   ├── eval_reduced.py            # 使用特征选择方法
│   └── utils.py            # 辅助函数
│
├── eval_baseline.ipynb               # Jupyter notebooks，实验笔记和可视化
├── baseline_process.py               # baseline 实验主函数
├── config.py               # 路径及参数
├── utils.py               # 辅助函数
├── README.md               # 项目介绍及运行说明
└── LICENSE                 # 许可证
```

---

## 📚 参考资料

- 论文原文: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095741742401385X)
- 作者代码仓库: [sbcblab/MOO-HFS](https://github.com/sbcblab/MOO-HFS)

---

## ✍️ 致谢

感谢课程老师、组员，以及原作者团队的开源贡献。

---

> 如有问题或建议，欢迎提 issue ！
