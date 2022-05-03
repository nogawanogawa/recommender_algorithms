import pandas as pd
import os

def user(homedir='/home') -> pd.DataFrame:
    cols = ['use_id', 'age', 'gender', 'occupation', 'zip_code']
    df = pd.read_csv(os.path.join(homedir, 'ml-100k/u.user'), sep='|', header=None, names=cols)
    return df

def data(homedir='/home') -> pd.DataFrame:
    cols = ['use_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(os.path.join(homedir, 'ml-100k/u.data'), sep='\t', header=None, names=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def genre(homedir='/home') -> pd.Series:
    cols = ['genre_name', 'id']
    df = pd.read_csv(os.path.join(homedir, 'ml-100k/u.genre'), sep='|', header=None, names=cols)
    df = df.sort_values('id')
    return df['genre_name']

def item(homedir='/home') -> pd.DataFrame:
    genre_names = genre(homedir).values.tolist()
    cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb URL']
    cols.extend(genre_names)

    return pd.read_csv(os.path.join(homedir, 'ml-100k/u.item'), sep='|', header=None, names=cols, encoding="ISO-8859-1")

if __name__ == "__main__":
    print(item())