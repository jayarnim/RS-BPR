import torch
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class NeuralCollaborativeFiltering(BaseModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Multi-Layer Perceptron (MLP),
        MLP & id embedding based latent factor model,
        sub-module of Neural Matrix Factorization (NeuMF)
        to learn high-rank nonlinear user-item extracteds.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
            hidden_dim (list):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__(locals())
        
        self.pred_dim = hidden_dim[-1]

        # IDX EMBEDDING ==========
        self.embedding = build_embedding_layer(
            name="idx",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # MATCHING FUNCTION LEARNING ==========
        self.matching = build_matching_layer(
            name="ncf",
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
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
        # MATCHING FUNCTION LEARNING ==========
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