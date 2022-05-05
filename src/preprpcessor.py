import dataloader 
import pandas as pd
import sklearn.preprocessing as sp

def preprocess(df, user, item):
    enc = sp.OneHotEncoder(sparse=False)
    
    user = pd.concat([user[["user_id", "age"]], 
                    pd.DataFrame(enc.fit_transform(user[["gender", "occupation"]]))
                    ], axis=1)

    df = pd.merge(df, user, on="user_id")

    df = df.drop(['user_id', 'item_id', 'timestamp'], axis=1)

    return df 

if __name__ == '__main__':
    df = dataloader.data()
    user = dataloader.user()
    item = dataloader.item()
    df = preprocess(df, user, item)
    print(df)
