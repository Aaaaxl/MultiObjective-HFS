import yaml
from argparse import Namespace


def load_config(path: str = "config.yaml") -> Namespace:
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 类型转换
    for key in ("min_features", "max_features", "cv_k", "n_max", "n_gen", "pop_size"):
        if key in cfg:
            cfg[key] = int(cfg[key])

    for key in ("fs_prob"):
        if key in cfg:
            cfg[key] = float(cfg[key])

    return Namespace(**cfg)
