import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import yaml
import re
import joblib

import random, numpy as np, torch


def set_seed(seed=42):
    """
    定义一个随机数种子,以确保每次训练是可复现的
    """
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # Numpy 随机数
    torch.manual_seed(seed)  # CPU 上的 PyTorch 随机数
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 上的 PyTorch 随机数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse_loss(
    input, target, masked_value="NA", reduction="mean"
):  # 定义一个带掩码的均方误差(MSE)
    if masked_value == "NA":
        mask = (
            target.isnan()
        )  # 如果缺失值用NaN表示，则勇敢函数target.isnan找出缺失的位置
    else:
        mask = target == masked_value  # 否则的话就用设定的缺失值的表示符号表示缺失值
    out = (input[~mask] - target[~mask]) ** 2  # 只使用非缺失值的部分计算MSE
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


def load_config(
    config_path: str = "config/data_conf.yml",
) -> Dict[str, Any]:  # 加载配置文件config.yml
    """Load configuration"""
    config_path_obj = Path(config_path)
    return yaml.safe_load(config_path_obj.read_text(encoding="utf-8"))


def extract_config(
    run_df: pd.DataFrame, run_id: str
) -> Dict[str, Any]:  # 从mlflow导出的run_df中找到指定的run_id对应的超参字段
    """Extract the config dict from mlflow run_df filtered by run_id.

    Args:
        run_df (pd.DataFrame): mlflow run_df
        run_id (str): mlflow run_id

    Returns:
        Dict: the config used to train the model associated with run_id
    """
    # mlflow是一个开源工具箱,其功能在于实验跟踪,可以在训练时把超参数,指标和模型文件记录下来
    # 在mlflow中,训练会话叫做run,每个run有一个唯一的run_id
    # run_df是mlflow的查询API,能把一批run的数据拉出来,其中包含run_id,参数,指标,状态等等

    run_info = (
        run_df.query("run_id == @run_id")
        .filter(regex="^params\.", axis=1)
        .reset_index(drop=True)
    )
    run_info_dict = run_info.loc[0].to_dict()
    new_dict = dict()
    for k, v in run_info_dict.items():
        k = k.replace("params.", "")
        if v == "None":
            v = None
        elif v == "True":
            v = True
        elif v == "False":
            v = False
        elif re.search(
            r"(^\d+$)|(^[\d.]+$)|(^[\d.]+e-?[\d.]+$)|(^\[[\d.,e\- ]+\]$)", str(v)
        ):
            v = json.loads(str(v))
        if k != "betas":
            new_dict[k] = v
        else:
            match = re.match(r"\(([\d.]+),\s*([\d.]+)\)", v)
            if match:
                new_dict["adam_beta1"] = float(match.group(1))
                new_dict["adam_beta2"] = float(match.group(2))

    return new_dict


class TargetZScaler:
    """
    只对一个连续标量目标做 z-score 标准化：
        z = (y - mean) / std
    说明：当前仅用于训练阶段，不依赖反标准化。
    """

    def __init__(self, mean: float = None, std: float = None, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        self.mean = float(np.nanmean(y))
        self.std = float(np.nanstd(y) + self.eps)
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=float)
        return (y - self.mean) / self.std

    def save(self, path: str):
        joblib.dump({"mean": self.mean, "std": self.std, "eps": self.eps}, path)

    @staticmethod
    def load(path: str) -> "TargetZScaler":
        obj = joblib.load(path)
        return TargetZScaler(mean=obj["mean"], std=obj["std"], eps=obj.get("eps", 1e-8))
