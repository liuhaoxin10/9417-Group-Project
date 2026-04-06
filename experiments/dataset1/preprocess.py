
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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


def save_processed_splits(dataset_name, X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame(X_train).to_csv(f"data/processed/{dataset_name}_X_train.csv", index=False)
    pd.DataFrame(X_val).to_csv(f"data/processed/{dataset_name}_X_val.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"data/processed/{dataset_name}_X_test.csv", index=False)

    pd.DataFrame(np.array(y_train), columns=["target"]).to_csv(
        f"data/processed/{dataset_name}_y_train.csv", index=False
    )
    pd.DataFrame(np.array(y_val), columns=["target"]).to_csv(
        f"data/processed/{dataset_name}_y_val.csv", index=False
    )
    pd.DataFrame(np.array(y_test), columns=["target"]).to_csv(
        f"data/processed/{dataset_name}_y_test.csv", index=False
    )


def save_processed_full(dataset_name, X_processed, y_processed):
    os.makedirs("data/processed", exist_ok=True)

    df_X = pd.DataFrame(X_processed)
    df_y = pd.DataFrame(np.array(y_processed), columns=["target"])
    df_full = pd.concat([df_X, df_y], axis=1)
    df_full.to_csv(f"data/processed/{dataset_name}_processed_full.csv", index=False)


def prepare_data(df, target_col, task_type="classification", drop_cols=None,
                 dataset_name=None, save_processed=True):
    df = df.copy()

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        print("当前可用列名：", list(df.columns))
        raise ValueError(f"target列 '{target_col}' 不在数据中，请检查列名。")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if task_type == "classification":
        y, label_encoder = encode_target_if_needed(y)
    else:
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.median())
        label_encoder = None

    preprocessor = build_preprocessor(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if task_type == "classification" else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full if task_type == "classification" else None
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    X_all_processed = preprocessor.transform(X)
    X_all_processed = np.asarray(X_all_processed, dtype=np.float32)
    y_all_processed = np.array(y)

    if save_processed and dataset_name is not None:
        save_processed_splits(
            dataset_name=dataset_name,
            X_train=X_train_processed,
            X_val=X_val_processed,
            X_test=X_test_processed,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
        save_processed_full(
            dataset_name=dataset_name,
            X_processed=X_all_processed,
            y_processed=y_all_processed
        )

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": label_encoder
    }


def load_wine_data():
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df


def load_divorce_data():
    df = pd.read_csv("data/divorce.csv", sep=";")
    df.columns = df.columns.str.strip()
    return df


def load_german_credit_data():
    df = pd.read_csv(
        "data/german.data",
        sep=r"\s+",
        header=None,
        engine="python"
    )
    df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
    return df


def load_bike_data():
    df = pd.read_csv("data/day.csv")
    df.columns = df.columns.str.strip()

    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"])
        df["year_extracted"] = df["dteday"].dt.year
        df["month_extracted"] = df["dteday"].dt.month
        df["day_extracted"] = df["dteday"].dt.day
        df["weekday_extracted"] = df["dteday"].dt.weekday
        df = df.drop(columns=["dteday"])

    return df


def load_appliances_data():
    df = pd.read_csv("data/energydata_complete.csv")
    df.columns = df.columns.str.strip()
    df = df.sample(n=10000, random_state=42)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["weekday"] = df["date"].dt.weekday
        df = df.drop(columns=["date"])

    return df


datasets = [
    {"name": "wine", "loader": load_wine_data, "target": "target", "task": "classification", "drop_cols": None},
    {"name": "divorce", "loader": load_divorce_data, "target": "Class", "task": "classification", "drop_cols": None},
    {"name": "german_credit", "loader": load_german_credit_data, "target": "target", "task": "classification", "drop_cols": None},
    {"name": "bike_sharing", "loader": load_bike_data, "target": "cnt", "task": "regression", "drop_cols": ["casual", "registered"]},
    {"name": "appliances_energy", "loader": load_appliances_data, "target": "Appliances", "task": "regression", "drop_cols": None}
]


if __name__ == "__main__":
    for cfg in datasets:
        print(f"\\n===== Preprocessing {cfg['name']} =====")

        df = cfg["loader"]()
        print("raw shape:", df.shape)

        if cfg["name"] == "divorce" and cfg["target"] not in df.columns:
            possible_targets = [col for col in df.columns if col.lower() in ["class", "target", "label"]]
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
            save_processed=True
        )

        print("X_train shape:", data_dict["X_train"].shape)
        print("X_val shape:", data_dict["X_val"].shape)
        print("X_test shape:", data_dict["X_test"].shape)

    print("\\nAll preprocessing finished.")
