# 🌟 MultiObjective-HFS

本仓库收录了《计算智能》课程第7组阅读文献复现并改进相关算法的全部代码。

> **原文献**：[《Enhancing classification with hybrid feature selection: A multi-objective genetic algorithm for high-dimensional data》](https://www.sciencedirect.com/science/article/pii/S095741742401385X)  
> **作者源码**：[sbcblab/MOO-HFS](https://github.com/sbcblab/MOO-HFS)

---

## ✅ 当前实现的功能

- [√] 多目标遗传算法 (MOGA) 的主要模块实现
- [x] 论文中混合特征选择（过滤法+包装法）逻辑复现
- [x] 支持高维数据集的特征子集搜索与评估
- [x] 部分核心指标与算法实验复现

---

## 🛠️ 已进行的改进

- [ ] 优化代码结构，增强可读性与复用性
- [ ] 提升部分算法的运行效率
- [ ] 增加详细注释与说明文档
- [ ] 其它创新点请在此补充...

---

## 📁 目录结构

```
MOO-HFS/
│
├── initial_feature_selection/                   # 特征选择部分
│   ├── data/                # 存储处理后的数据
│   │   ├── Arrhythmia/
│   │   ├── Breast_Cancer/
│   │   ├── Leukemia/
│   │   ├── p53_Mutants/
│   │   └── importances/
│   │
│   ├── data_process/          # 处理数据代码部分
│   │   ├── arrhythmia/
│   │   │   ├── raw_data_process.ipynb 
│   │   │   └── arrhythmia_des.md
│   │   │
│   │   ├── breast_cancer/
│   │   │   ├── raw_data_preprocess.ipynb
│   │   │   ├── merge_data_label.ipynb
│   │   │   ├── data_check.ipynb
│   │   │   └── breast_des.md
│   │   │
│   │   ├── leukemia/
│   │   │   ├──
│   │   │   ├──
│   │   │   └──
│   │   │ 
│   │   ├── p53_mutants/
│   │   │   ├── raw_data_process.ipynb
│   │   │   └── p53_des.md
│   │ 
│   ├── deterministic/          # 确定性特征选择代码
│   │   ├── anovafvalue.py
│   │   ├── kruskalwallis.py
│   │   ├── lassocv.py
│   │   ├── mrmr.py
│   │   └── relieff.py
│   │
│   ├── non_deterministic/          # 非确定性特征选择代码
│   │   ├── decisiontree.py
│   │   ├── linearsvm.py
│   │   ├── mutualinfo.py
│   │   └── randomforest.py
│   │
│   ├── feature_selection.ipynb/
│   ├── config.py
│   ├── feature_selection.py
│   └── utils.py
│
├── src/                    # 源代码目录
│   ├── feature_selection.py # 主要的特征选择算法实现
│   ├── utils.py            # 工具函数，如数据加载、评估指标等
│   └── main.py             # 项目入口，训练和测试脚本
│
├── experiments/            # 实验相关脚本或配置文件
│   └── experiment1.py      # 某个具体实验脚本
│
├── notebooks/              # Jupyter notebooks，实验笔记和可视化
│
├── requirements.txt        # 依赖包列表
├── README.md               # 项目介绍及运行说明
├── LICENSE                 # 许可证
└── setup.py                # （可选）安装脚本
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
