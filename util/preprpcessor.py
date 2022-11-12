from util.dataloader import Data
import pandas as pd
import sklearn.preprocessing as sp
from sklearn.preprocessing import OneHotEncoder


class Preprocessor:
    def __init__(self) -> None:
        self.enc_user = OneHotEncoder(sparse=False)
        self.enc_item = OneHotEncoder(sparse=False)

    def fit(self, df, user, item):
        self.enc_user.fit(user[["user_id", "gender", "occupation"]])
        self.enc_item.fit(item[["item_id"]])

        return self.trenasform(df, user, item)

    def trenasform(self, df, user, item):
        # ユーザーの特徴量でカテゴリ変数があるのでone-hotに直す
        user = pd.concat(
            [
                user[["age", "user_id"]],
                pd.DataFrame(self.enc_user.transform(user[["user_id", "gender", "occupation"]])),
            ],
            axis=1,
        )
        df = pd.merge(df, user, on="user_id")

        item = pd.concat(
            [
                item[[
                    "item_id",
                    "unknown",
                    "Action",
                    "Adventure",
                    "Animation",
                    "Children's",
                    "Comedy",
                    "Crime",
                    "Documentary",
                    "Drama",
                    "Fantasy",
                    "Film-Noir",
                    "Horror",
                    "Musical",
                    "Mystery",
                    "Romance",
                    "Sci-Fi",
                    "Thriller",
                    "War",
                    "Western",
                ]],
                pd.DataFrame(self.enc_item.transform(item[["item_id"]])),
            ],
            axis=1,
        )
        df = pd.merge(
            df, item, on="item_id",
        )

        df = df.drop(["user_id", "item_id", "timestamp"], axis=1)

        return df



if __name__ == "__main__":
    dataloader = Data()
    df = dataloader.data()
    user = dataloader.user()
    item = dataloader.item()
    df = Preprocessor().preprocess(df, user, item)
    print(df)
