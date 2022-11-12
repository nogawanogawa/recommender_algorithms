import random

import implicit
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse

import dataloader

if __name__ == '__main__':
    train_df = dataloader.data(test=False)
    test_df = dataloader.data(test=True)
    user_df = dataloader.user()
    item_df = dataloader.item()

    train_df = train_df[["user_id", "item_id", "rating"]]

    users = train_df["user_id"].unique()
    movies = train_df["item_id"].unique()
    shape = (len(users), len(movies))

    # Create indices for users and movies
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    movie_cat = CategoricalDtype(categories=sorted(movies), ordered=True)
    user_index = train_df["user_id"].astype(user_cat).cat.codes
    movie_index = train_df["item_id"].astype(movie_cat).cat.codes

    # Conversion via COO matrix
    coo = sparse.coo_matrix((train_df["rating"], (user_index, movie_index)), shape=shape)
    csr = coo.tocsr()

    print(csr)

    data = sparse.csr_matrix(train_df.values)
    model = implicit.als.AlternatingLeastSquares(factors=64)

    # train
    model.fit(csr)

    # recommend items for a user
    recommendations = model.recommend(0, data[0])

    print(recommendations)
    # find related items
    related = model.similar_items(0)
    print(related)
