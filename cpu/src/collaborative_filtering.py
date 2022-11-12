import pandas as pd
import numpy as np
from util.dataloader import Data
from typing import Tuple, List
from eval.eval_func import *

class CF:
    def __init__(self, rating: pd.DataFrame):
        # ratingをuser=id, item_idを使用した行列に変換する
        self.df = rating.pivot_table(index="user_id", columns="item_id", values="rating")

    def pearson_coefficient(
        self, df: pd.DataFrame, threshold: float = 0.2
    ) -> pd.DataFrame:
        """ユーザーに関する相関係数

        df : 列がitem, 行がuserを示すDataFrame
        threshold: 協調フィルタリングの計算に使用する際の相関係数の下限値
        """
        corr = df.T.corr()  # ピアソンの相関係数
        corr.values[
            [np.arange(corr.shape[0])], [np.arange(corr.shape[0])]
        ] = np.nan  # 対角線成分をnanに設定

        sim = corr.where(corr > threshold, np.nan)
        return sim

    def average_user_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """avg_ratingをアイテムの個数分だけ拡張する

        Args:
            df (pd.DataFrame): ユーザーが付与した実際のレーティング
            avg_rating (pd.Series): ユーザーごとの平均レーティング

        Returns:
            pd.DataFrame: avg_ratingをアイテムの個数分だけ拡張したDataFrame
        """

        avg_rating = df.mean(axis=1)

        # 横向きにavg_ratingをコピーして拡張
        S = pd.concat(
            [pd.DataFrame(avg_rating) for _ in range(df.shape[1] - 1)], axis=1
        )

        # カラムのindexを1~アイテム数に振り直す
        S = S.set_axis(np.arange(1, S.shape[1] + 1), axis=1, copy=False)

        return S

    def user_rating_diff(
        self, df: pd.DataFrame, avg_rating: pd.DataFrame
    ) -> pd.DataFrame:
        """ユーザーごとの平均に対するレーティング

        Args:
            df (pd.DataFrame): ユーザーが付与した実際のレーティング
            avg_rating (pd.Series): ユーザーごとの平均レーティング

        Returns:
            pd.DataFrame: ユーザーごとの平均に対するレーティング
        """

        # ユーザーごとのレーティングの平均値
        df = df.copy()
        df = df - avg_rating

        return df

    def sum_sim(self, sim: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """nullを含んだときの値のsimの総和の計算

        Args:
            sim (pd.DataFrame): 類似度を
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """

        # ユーザーが付与した実際のレーティングを、評価していれば1, 評価していなければ0に変換
        df = df.copy()
        df = df.where(~df.isna(), 1)
        df = df.fillna(0)

        # ユーザーの類似度と内積を取ることで、自分を除く類似したユーザーの類似度の合計を求める
        d = sim.fillna(0).dot(df.fillna(0))

        return d

    def train(self) -> None:
        """ 協調フィルタリングの計算 """
        df = self.df

        sim = self.pearson_coefficient(df)
        avg_user_rating = self.average_user_rating(df)
        y_ybar = self.user_rating_diff(df, avg_user_rating)

        d = self.sum_sim(sim, df)
        c = sim.fillna(0).dot(y_ybar.fillna(0))

        self.model = avg_user_rating.add(c / d)

        self.model = pd.melt(self.model.reset_index(), id_vars="user_id", value_vars=self.model.columns).rename(columns={"variable": "item_id"})
        self.model = self.model.set_index(["user_id", "item_id"])

        return

    def predict(self, test_df:pd.DataFrame) -> List[float]:
        """ あたえられたdfのすべてのratingを返す """
        return [self.predict_score(user_id, item_id) for user_id, item_id in test_df.values]


    def predict_score(self, user_id:int, item_id:int) -> float:
        """ user_id, item_id毎の推論したratingを返す """
        try:
            return self.model.loc[(user_id, item_id)]["value"]
        except:
            return np.nan


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load() # user_df, item_df, train_df, test_df

if __name__ == "__main__":

    user_df, item_df, train_df, test_df = load_data()

    # train
    cf = CF(train_df)
    cf.train()

    result = cf.predict(test_df[["user_id", "item_id"]])
    test_df["rating_predict"] = result
    test_df = test_df.dropna(subset=['rating_predict'])

    ####### Evaluate #######
    # RMSE
    # わかっているratingについて、推定したratingとのRMSEを取得する
    rmse_score = rmse(
        test_df["rating_predict"].values.tolist(), test_df["rating"].values.tolist()
    )
    print(f"rmse: {rmse_score}")

    # Recall
    # 推定ratingを上からk件とってきてそれのrecallを取る
    k = 10
    true_df = (
        test_df.sort_values(["user_id", "rating", "item_id"], ascending=False)
        .groupby("user_id")
        .head(k)
    )
    y_true = (
        true_df[["user_id", "item_id"]]
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    candidate = (
        test_df.sort_values(["user_id", "rating_predict", "item_id"], ascending=False)
        .groupby("user_id")
        .head(k)
    )
    y_pred = (
        candidate[["user_id", "item_id"]]
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )
    recall_socre = recall(y_pred, y_true)
    print(f"recall@{k}: {recall_socre}")

    # nDCG
    k = 10
    pred_df = test_df[["user_id", "item_id"]]

    pred_df = pred_df.merge(
        test_df[["user_id", "item_id", "rating_predict"]], on=["user_id", "item_id"], how="left"
    )
    pred_df = pred_df.sort_values(["user_id", "rating_predict", "item_id"], ascending=False)
    pred_df = (
        pred_df.reset_index(drop=True).reset_index().rename(columns={"index": "rank"})
    )
    ndcg_score = ndcg(y_pred=pred_df, y_true=test_df, k=k)
    print(f"ndcg@{k}: {ndcg_score}")
