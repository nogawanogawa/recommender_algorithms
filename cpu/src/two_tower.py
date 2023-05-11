from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from util.dataloader import Data
from util.preprpcessor import Preprocessor
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy

import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import copy
import pytorch_lightning as pl
from tqdm import tqdm
from model.two_tower_model import UserTower, ItemTower
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Parameters:
    max_epochs: int = 30
    early_stopping: bool = True
    early_stopping_patience: int = 5
    batch_size: int = 8192
    learning_rate: float = 0.01


params = Parameters()


class Model(pl.LightningModule):
    def __init__(self, user_emb_dim: int, item_emb_dim: int):
        """
        Args:
            n (int, optional): weight matrixのサイズ
            k (int, optional): 学習データの行数
        """
        super().__init__()
        user_ln, item_ln = [user_emb_dim, 1024, 512, 128], [
            item_emb_dim,
            1024,
            512,
            128,
        ]
        self.user_tower = UserTower(user_ln)
        self.item_tower = ItemTower(item_ln)

        self.dot = torch.matmul

        # Loss
        self.loss_fn = nn.BCEWithLogitsLoss()  # 本来はlossは自作しないとダメそう

        # コンソールログ出力用の変数
        self.log_outputs = {}

    def forward(self, user_x, item_x):
        out = self.dot(self.user_tower(user_x), self.item_tower(item_x).t())
        return out

    # 学習のstep内の処理
    def training_step(self, batch, batch_index):
        user_x, item_x, y = batch
        pred = self(user_x, item_x)
        loss = self.loss_fn(pred, y)
        return {"loss": loss}

    # validのstep内の処理
    def validation_step(self, batch, batch_index):
        user_x, item_x, y = batch
        pred = self(user_x, item_x)
        loss = self.loss_fn(pred, y)
        return {"val_loss": loss}

    # 学習の全バッチ終了時の処理(Lossの平均を出力)
    def training_epoch_end(self, outputs) -> None:
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_outputs["loss"] = train_loss
        return

    # validのepoch終了時の処理、ロスの集計などを行う
    def validation_epoch_end(self, outputs) -> None:
        valid_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log_dict({"valid_loss": valid_loss})
        self.log_outputs["valid_loss"] = valid_loss
        return

    # epoch開始時の処理
    def on_train_epoch_start(self) -> None:
        self.print(f"Epoch {self.trainer.current_epoch+1}/{self.trainer.max_epochs}")
        return super().on_train_epoch_start()

    # epoch終了時の処理
    def on_train_epoch_end(self) -> None:
        train_loss = self.log_outputs["loss"]
        valid_loss = self.log_outputs["valid_loss"]
        self.print(f"loss: {train_loss:.3f} - val_loss: {valid_loss:.3f}")
        return super().on_train_epoch_end()

    # 学習開始時の処理
    def on_train_start(self) -> None:
        self.print(f"Train start")
        return super().on_train_start()

    # 学習終了時の処理
    def on_train_end(self) -> None:
        self.print(f"Train end")
        return super().on_train_end()


# for reproducibility

callbacks = []

# early stoppingをするなら
if params.early_stopping:
    callbacks.append(
        EarlyStopping(monitor="valid_loss", patience=params.early_stopping_patience)
    )

# model 保存のcallbackを用意
checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    dirpath="./pl-checkpoints",
    filename="sample-{epoch:02d}-{valid_loss:.03f}-{valid_accuracy:.03f}",
    save_top_k=3,
    mode="min",
)
callbacks.append(checkpoint_callback)


class TwoTower:
    def __init__(self, user_df, item_df) -> None:
        self.model = None

    def user_emb(self, df, user_df, item_df):
        """userの特徴を取る
        - カテゴリごとの平均レーティング
            - 未評価は平均値埋め
        - 年齢(age)
        - 性別(gender)
        - 職種(occupation)

        Args:
            df (pd.DataFrame): _description_
            user_df (pd.DataFrame): _description_
        """
        df = df.copy()
        df = df.merge(user_df, on="user_id", how="left").merge(
            item_df, on="item_id", how="left"
        )

        category_average_rating_dict = df.groupby("")
        # この辺の特徴量エンジニアリングが必要

        return

    def item_emb(self, df, item_df):
        pass

    def train(self, train_df):
        # TODO: CVを取る
        # ref: https://github.com/Lightning-AI/lightning/blob/master/examples/pl_loops/kfold.py
        X_train, X_valid, y_train, y_valid = train_test_split(
            train_df.drop(columns=["rating"]),
            train_df[["rating"]],
            test_size=0.2,
            random_state=42,
        )

        self.user_emb(X_train, user_df)

        # validは距離の比較に使用する

    def predict(self, test_df):
        return


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load()  # user_df, item_df, train_df, test_df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 各dataframeからtraining用データを作成
    preprpcessor = Preprocessor()
    train_df = preprpcessor.fit(train_df, user_df, item_df)
    test_df = preprpcessor.trenasform(test_df, user_df, item_df)

    return train_df, test_df


if __name__ == "__main__":

    user_df, item_df, train_df, test_df = load_data()

    # 各dataframeからtraining用データを作成
    train_df, test_df = preprocess(train_df, test_df, user_df, item_df)

    X_train = train_df.drop(columns=["rating"]).fillna(0)
    X_test = test_df.drop(columns=["rating"]).fillna(0)
    y_train = train_df[["rating"]]
    y_test = test_df[["rating"]]

    # convert to 32-bit numbers to send to GPU
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    X_tensor = torch.from_numpy(X_train).to(device)
    y_tensor = torch.from_numpy(y_train).to(device)

    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    valid_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # LightningModuleをインスタンス化
    model = TwoTower(user_emb_dim, item_emb_dim)

    # DatasetはDataLoader経由でイテレーションする
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size)

    # Trainer定義
    trainer = pl.Trainer(
        max_epochs=params.max_epochs,
        callbacks=callbacks,
    )

    # トレーニング実行
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
