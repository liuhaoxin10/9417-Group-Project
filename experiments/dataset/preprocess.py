import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


RANDOM_STATE = 42
RAW_DIR = "data/raw"
LEGACY_DATA_DIR = "data"
PROCESSED_DIR = "data/processed"


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def resolve_data_path(filename):
    """
    优先从 data/raw/ 找；
    如果本地还没把原始数据搬进去，则回退到 data/ 找。
    这样既兼容当前本地状态，也和组员要求的长期结构兼容。
    """
    candidates = [
        os.path.join(RAW_DIR, filename),
        os.path.join(LEGACY_DATA_DIR, filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"找不到数据文件 {filename}。\n"
        f"已尝试这些路径：\n" +
        "\n".join(candidates) +
        "\n请把原始数据放到 data/raw/，或确认 data/ 下是否存在该文件。"
    )


def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def encode_target_if_needed(y):
    if y.dtype == "object" or str(y.dtype) == "category":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        return y_encoded, le
    return y.to_numpy(), None


def save_array_with_feature_names(path, array, feature_names):
    df = pd.DataFrame(array, columns=feature_names)
    df.to_csv(path, index=False)


def save_target(path, y):
    pd.DataFrame({"target": np.array(y)}).to_csv(path, index=False)


def save_feature_names(dataset_name, feature_names):
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        os.path.join(PROCESSED_DIR, f"{dataset_name}_feature_names.csv"),
        index=False
    )


def save_metadata(dataset_name, task_type, target_col, raw_path, original_features,
                  processed_features, X_train, X_val, X_test, extra_info=None):
    metadata = {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target": target_col,
        "raw_data_path": raw_path,
        "processed_data_dir": PROCESSED_DIR,
        "original_features": list(original_features),
        "processed_features": list(processed_features),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "saved_files": [
            f"{dataset_name}_X_train.csv",
            f"{dataset_name}_X_val.csv",
            f"{dataset_name}_X_test.csv",
            f"{dataset_name}_y_train.csv",
            f"{dataset_name}_y_val.csv",
            f"{dataset_name}_y_test.csv",
            f"{dataset_name}_feature_names.csv",
            f"{dataset_name}_metadata.json",
            f"{dataset_name}_preprocessor.joblib"
        ]
    }

    if extra_info is not None:
        metadata["extra_info"] = extra_info

    with open(
        os.path.join(PROCESSED_DIR, f"{dataset_name}_metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def prepare_data(df, target_col, task_type="classification", drop_cols=None,
                 dataset_name=None, raw_path=None):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if target_col not in df.columns:
        print("当前可用列名：", list(df.columns))
        raise ValueError(f"target列 '{target_col}' 不在数据中，请检查列名。")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    original_features = X.columns.tolist()

    if task_type == "classification":
        y, label_encoder = encode_target_if_needed(y)
    else:
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.median())
        y = y.to_numpy()
        label_encoder = None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if task_type == "classification" else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full if task_type == "classification" else None
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    save_array_with_feature_names(
        os.path.join(PROCESSED_DIR, f"{dataset_name}_X_train.csv"),
        X_train_processed,
        feature_names
    )
    save_array_with_feature_names(
        os.path.join(PROCESSED_DIR, f"{dataset_name}_X_val.csv"),
        X_val_processed,
        feature_names
    )
    save_array_with_feature_names(
        os.path.join(PROCESSED_DIR, f"{dataset_name}_X_test.csv"),
        X_test_processed,
        feature_names
    )

    save_target(os.path.join(PROCESSED_DIR, f"{dataset_name}_y_train.csv"), y_train)
    save_target(os.path.join(PROCESSED_DIR, f"{dataset_name}_y_val.csv"), y_val)
    save_target(os.path.join(PROCESSED_DIR, f"{dataset_name}_y_test.csv"), y_test)

    save_feature_names(dataset_name, feature_names)
    save_metadata(
        dataset_name=dataset_name,
        task_type=task_type,
        target_col=target_col,
        raw_path=raw_path,
        original_features=original_features,
        processed_features=feature_names,
        X_train=X_train_processed,
        X_val=X_val_processed,
        X_test=X_test_processed
    )

    joblib.dump(
        preprocessor,
        os.path.join(PROCESSED_DIR, f"{dataset_name}_preprocessor.joblib")
    )

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder
    }


def load_wine_data():
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df, "sklearn.datasets.load_wine()"


def load_divorce_data():
    path = resolve_data_path("divorce.csv")
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df, path


def load_german_credit_data():
    path = resolve_data_path("german.data")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python"
    )
    df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
    return df, path


def load_bike_data():
    path = resolve_data_path("day.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # 按组员意见：删掉 dteday，不再生成重复日期特征
    if "dteday" in df.columns:
        df = df.drop(columns=["dteday"])

    return df, path


def load_appliances_data():
    path = resolve_data_path("energydata_complete.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # 按组员意见：不采样到 10000，直接使用全量，确保 n > 10000
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["weekday"] = df["date"].dt.weekday
        df = df.drop(columns=["date"])

    return df, path


datasets = [
    {
        "name": "wine",
        "loader": load_wine_data,
        "target": "target",
        "task": "classification",
        "drop_cols": None
    },
    {
        "name": "divorce",
        "loader": load_divorce_data,
        "target": "Class",
        "task": "classification",
        "drop_cols": None
    },
    {
        "name": "german_credit",
        "loader": load_german_credit_data,
        "target": "target",
        "task": "classification",
        "drop_cols": None
    },
    {
        "name": "bike_sharing",
        "loader": load_bike_data,
        "target": "cnt",
        "task": "regression",
        "drop_cols": ["instant", "casual", "registered"]
    },
    {
        "name": "appliances_energy",
        "loader": load_appliances_data,
        "target": "Appliances",
        "task": "regression",
        "drop_cols": None
    }
]


if __name__ == "__main__":
    ensure_dirs()

    for cfg in datasets:
        print(f"\n===== Preprocessing {cfg['name']} =====")

        df, raw_path = cfg["loader"]()
        print("raw shape:", df.shape)

        if cfg["name"] == "divorce" and cfg["target"] not in df.columns:
            possible_targets = [
                col for col in df.columns
                if col.lower() in ["class", "target", "label"]
            ]
            if possible_targets:
                cfg["target"] = possible_targets[0]
            else:
                print("Divorce 数据集列名如下：")
                print(list(df.columns))
                raise ValueError("请手动确认 divorce 的标签列名。")

        data_dict = prepare_data(
            df=df,
            target_col=cfg["target"],
            task_type=cfg["task"],
            drop_cols=cfg["drop_cols"],
            dataset_name=cfg["name"],
            raw_path=raw_path
        )

        print(
            "processed train/val/test:",
            data_dict["X_train"].shape,
            data_dict["X_val"].shape,
            data_dict["X_test"].shape
        )
        print("processed feature count:", len(data_dict["feature_names"]))

    print("\nAll preprocessing finished.")