import pandas as pd
import numpy as np
import dataloader

class CF:
    def __init__ (self, rating:pd.DataFrame):
        self.df = rating
    
    def pearson_coefficient(self, df: pd.DataFrame, threshold=0.2) -> pd.DataFrame:
        """ユーザーに関する相関係数
        
        df : 列がitem, 行がuserを示すDataFrame
        threshold: 協調フィルタリングの計算に使用する際の相関係数の下限値
        """
        corr = df.T.corr() #ピアソンの相関係数
        corr.values[[np.arange(corr.shape[0])], [np.arange(corr.shape[0])]]= np.nan #対角線成分をnanに設定

        sim = corr.where(corr > threshold, np.nan)
        return sim 

    def average_user_rating(self, df:pd.DataFrame) -> pd.DataFrame:
        """avg_ratingをアイテムの個数分だけ拡張する

        Args:
            df (pd.DataFrame): ユーザーが付与した実際のレーティング
            avg_rating (pd.Series): ユーザーごとの平均レーティング

        Returns:
            pd.DataFrame: avg_ratingをアイテムの個数分だけ拡張したDataFrame
        """

        avg_rating = df.T.mean()
        S = pd.DataFrame(avg_rating)

        for i in range(df.shape[1] -1):
            S = pd.concat([S, pd.DataFrame(avg_rating)], axis=1)

        # カラムのindexを1~アイテム数に振り直す
        S = S.T.reset_index(drop = True)
        S.index = np.arange(1, len(S)+1)
        S = S.T
        return S

    def user_rating_diff(self, df:pd.DataFrame) -> pd.DataFrame:
        """ユーザーごとの平均に対するレーティング

        Args:
            df (pd.DataFrame): ユーザーが付与した実際のレーティング
            avg_rating (pd.Series): ユーザーごとの平均レーティング

        Returns:
            pd.DataFrame: ユーザーごとの平均に対するレーティング
        """

        # ユーザーごとのレーティングの平均値
        avg_rating = df.T.mean()
        y_ybar = df.copy()

        for i in range(df.shape[1]):
            y_ybar.loc[:, i+1] = y_ybar.loc[:, i+1] - avg_rating

        return y_ybar
    
    def weight_sum(self, sim:pd.DataFrame, df: pd.DataFrame):
        """荷重和を取る

        Args:
            sim (pd.DataFrame): 類似度を
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        
        # ユーザーが付与した実際のレーティングを、評価していれば1, 評価していなければ0に変換 
        df_ = df.where(df.isna(), 1)
        df_ = df_.fillna(0)

        # ユーザーの類似度と内積を取ることで、自分を除く類似したユーザーの類似度の合計を求める
        d = sim.fillna(0).dot(df_.fillna(0))
        return d

    def run(self):
        """協調フィルタリングの計算

        Returns:
            _type_: _description_
        """

        df = self.df
        
        sim = self.pearson_coefficient(df)
        avg_user_rating = self.average_user_rating(df)
        y_ybar = self.user_rating_diff(df)

        d = self.weight_sum(sim, df)
        c = sim.fillna(0).dot(y_ybar.fillna(0))
        
        return avg_user_rating.add((c/d))

if __name__ == '__main__':
    df = dataloader.data()
    df = df.pivot_table(index='user_id', columns='item_id', values='rating')
    
    cf = CF(df)
    result = cf.run()
    print(result)
