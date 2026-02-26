import torch
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class GeneralMatrixFactorization(BaseModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Generalized Matrix Factorization (GMF),
        MF & id embedding based latent factor model,
        sub-module of Neural Matrix Factorization (NeuMF)
        to learn low-rank linear representation.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
        """
        super().__init__(locals())
        
        self.pred_dim = embedding_dim

        # IDX EMBEDDING ==========
        self.embedding = build_embedding_layer(
            name="idx",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # BILINEAR MATCHING FUNCTION ==========
        self.matching = build_matching_layer(
            name="mf",
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # IDX EMBEDDING ==========
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        # BILINEAR MATCHING FUNCTION ==========
        X_pred = self.matching(user_emb, item_emb)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit