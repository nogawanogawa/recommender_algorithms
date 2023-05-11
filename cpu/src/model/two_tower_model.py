from typing import List

import torch
from torch import nn


class UserTower(nn.Module):
    def __init__(
        self,
        ln: List[int],  # layerの次元をListでとってきている
    ):
        super(UserTower, self).__init__()

        self.item_layers = nn.Sequential(
            nn.Embedding(embedding_dim=ln[0]),
            nn.Linear(ln[0], ln[1], bias=True),
            nn.ReLU(),
            nn.Linear(ln[1], ln[2], bias=True),
            nn.ReLU(),
            nn.Linear(ln[2], ln[3], bias=True),
        )

    def forward(self, users):

        user_emb = self.user_emb(users)
        user_emb = self.user_layers(user_emb)

        return user_emb


class ItemTower(nn.Module):
    def __init__(self, n_items: int, ln: List[int]):  # layerの次元をarrayでとってきている
        super(ItemTower, self).__init__()

        self.item_layers = nn.Sequential(
            nn.Embedding(embedding_dim=ln[0]),
            nn.Linear(ln[0], ln[1], bias=True),
            nn.ReLU(),
            nn.Linear(ln[1], ln[2], bias=True),
            nn.ReLU(),
            nn.Linear(ln[2], ln[3], bias=True),
        )

    def forward(self, items):

        item_emb = self.item_emb(items)
        item_emb = self.item_layers(item_emb)

        return item_emb
