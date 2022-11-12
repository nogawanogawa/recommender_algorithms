import os
from typing import Tuple

import pandas as pd


class Data:
    def __init__(self) -> None:
        self.homedir = "/home"

    def user(self) -> pd.DataFrame:
        """userのデータの取得"""
        cols = ["user_id", "age", "gender", "occupation", "zip_code"]
        df = pd.read_csv(
            os.path.join(self.homedir, "ml-100k/u.user"),
            sep="|",
            header=None,
            names=cols,
        )
        return df

    def item(self) -> pd.DataFrame:
        """itemのデータ取得（全量）"""
        genre_names = self.genre().values.tolist()
        cols = [
            "item_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb URL",
        ]
        cols.extend(genre_names)

        return pd.read_csv(
            os.path.join(self.homedir, "ml-100k/u.item"),
            sep="|",
            header=None,
            names=cols,
            encoding="ISO-8859-1",
        )

    def genre(self) -> pd.Series:
        cols = ["genre_name", "id"]
        df = pd.read_csv(
            os.path.join(self.homedir, "ml-100k/u.genre"),
            sep="|",
            header=None,
            names=cols,
        )
        df = df.sort_values("id")
        return df["genre_name"]

    def data(self, is_test: bool = False) -> pd.DataFrame:
        """ ratingの取得(test=Trueのときtest setを取得)"""
        cols = ["user_id", "item_id", "rating", "timestamp"]

        if is_test:
            df = pd.read_csv(
                os.path.join(self.homedir, "ml-100k/u1.test"),
                sep="\t",
                header=None,
                names=cols,
            )
        else:
            df = pd.read_csv(
                os.path.join(self.homedir, "ml-100k/u1.base"),
                sep="\t",
                header=None,
                names=cols,
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        return df

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        user_df = self.user()
        item_df = self.item()
        train_df = self.data(is_test=False)
        test_df = self.data(is_test=True)
        return user_df, item_df, train_df, test_df


if __name__ == "__main__":
    dl = Data()
    print(dl.genre())
