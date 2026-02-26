import torch
from components.base import BaseModel
from .gmf import GeneralMatrixFactorization
from .ncf import NeuralCollaborativeFiltering
from .layers.fusion import ConcatenationLayer
from .layers.prediction import ProjectionLayer


class NeuralMatrixFactorization(BaseModel):
    def __init__(
        self,
        gmf: GeneralMatrixFactorization,
        mlp: NeuralCollaborativeFiltering,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Neural Matrix Factorization (NeuMF),
        MF, MLP & id embedding based latent factor model,
        combining a Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP)
        to learn low-rank linear representation & high-rank nonlinear user-item extracteds.

        Args:
            gmf (nn.Module)
            mlp (nn.Moudle)
        """
        super().__init__(locals())

        # ENSEMBLE MODULES ==========
        self.gmf = gmf
        self.mlp = mlp
        self.pred_dim = gmf.pred_dim + mlp.pred_dim

        # FUSION ==========
        self.fusion = ConcatenationLayer()

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # ENSEMBLE LEARNING ==========
        args = (
            self.gmf(user_idx, item_idx),
            self.mlp(user_idx, item_idx),
        )
        # ENSEMBLE AGGREGATION ==========
        X_pred = self.fusion(*args)
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