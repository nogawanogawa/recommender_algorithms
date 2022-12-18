import copy
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from util.preprpcessor import Preprocessor
import pytorch_lightning as pl
import scipy
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin

# from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from eval.eval_func import *
from util.dataloader import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Parameters:
    max_epochs: int = 30
    early_stopping: bool = True
    early_stopping_patience: int = 10
    batch_size: int = 32
    learning_rate: float = 0.01

params = Parameters()


class Model(pl.LightningModule):
    """ Factorization Machine """

    def __init__(self, n: int = None, k: int = None):
        """
        Args:
            n (int, optional): weight matrixのサイズ
            k (int, optional): 学習データの行数
        """
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

        # Loss
        self.loss_fn = nn.MSELoss()

        # コンソールログ出力用の変数
        self.log_outputs = {}

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer

    # 学習のstep内の処理
    def training_step(self, batch, batch_index):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        return {"loss": loss}

    # validのstep内の処理
    def validation_step(self, batch, batch_index):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        return {"val_loss": loss}

   # predictのstep内の処理
    def predict_step(self, batch, batch_index):
        X, y = batch
        pred = self(X)
        return pred

    # 学習の全バッチ終了時の処理
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

class FactorizationMachine():
    def __init__(self) -> None:
        pass

    def train(self, train_df) -> None:

        # TODO: CVを取る
        # ref: https://github.com/Lightning-AI/lightning/blob/master/examples/pl_loops/kfold.py
        X_train, X_valid, y_train, y_valid = train_test_split(
            train_df.drop(columns=["rating"]), train_df[["rating"]], test_size=0.2, random_state=42
        )

        # convert to 32-bit numbers to send to GPU
        X_train = X_train.values.astype(np.float32)
        y_train = y_train.values.astype(np.float32)
        X_valid = X_valid.values.astype(np.float32)
        y_valid = y_valid.values.astype(np.float32)

        X_train = torch.from_numpy(X_train).to(device)
        y_train = torch.from_numpy(y_train).to(device)
        X_valid = torch.from_numpy(X_valid).to(device)
        y_valid = torch.from_numpy(y_valid).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)

        # DatasetはDataLoader経由でイテレーションする
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size)

        # LightningModuleをインスタンス化
        self.model = Model(n=X_train.shape[1], k=2048)

        # Trainer定義
        trainer = pl.Trainer(
            max_epochs=params.max_epochs,
            callbacks=callbacks,
        )

        # トレーニング実行
        trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


    def predict(self, test_df:pd.DataFrame) -> List[float]:
        """ あたえられたdfのすべてのratingを返す """

        test = test_df.reset_index(drop=True)

        X_test = test.drop(columns=["rating"]).fillna(0)
        y_test = test[["rating"]]

        # convert to 32-bit numbers to send to GPU
        X_test = X_test.values.astype(np.float32)
        y_test = y_test.values.astype(np.float32)

        X_test = torch.from_numpy(X_test).to(device)
        y_test = torch.from_numpy(y_test).to(device)

        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # DatasetはDataLoader経由でイテレーションする
        test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size)

        self.model.eval()
        all_preds = np.array([])
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                pred = self.model.predict_step(batch, batch_idx).detach().cpu().numpy()
                all_preds = np.append(all_preds, pred)

        return all_preds

# for reproducibility
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

    # TODO: preprocessが多分データ不足
    train_df, _test_df = preprocess(train_df, test_df, user_df, item_df)

    fm = FactorizationMachine()
    fm.train(train_df)

    # predict
    pred_df = test_df.copy()
    pred_df["y_pred"] = fm.predict(_test_df)
    pred_df = pred_df.dropna(subset=['y_pred'])

    pred_df["predicted_rank"] = pred_df.groupby(["user_id"])["y_pred"].rank(ascending=False, method='first')
    pred_df = pred_df[["user_id", "item_id", "y_pred", "predicted_rank"]]

    true_df = test_df.copy()
    true_df = true_df[["user_id", "item_id", "rating"]].rename(columns={"rating":"y_true"})
    true_df["optimal_rank"] = true_df.groupby(["user_id"])["y_true"].rank(ascending=False, method='first')
    true_df = true_df[["user_id", "item_id", "y_true", "optimal_rank"]]

    result = eval(predict_recom=pred_df, true_recom=true_df)
    print(result)

