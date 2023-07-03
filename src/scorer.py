from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Scorer(ABC):
    @abstractmethod
    def score(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        pass


class CosineSimilarityScorer(Scorer):
    """Calulate cosine similarity between template(1, C) and roi(N, C) features."""

    def check_shape(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> None:
        assert template_features.dim() == 2
        assert roi_features.dim() == 2
        assert template_features.size(1) == roi_features.size(1)
        assert len(template_features) == 1

    def score(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        if len(template_features.shape) == 1:
            template_features = template_features.unsqueeze(0)
        self.check_shape(template_features, roi_features)

        template_features = F.normalize(template_features, dim=1)
        roi_features = F.normalize(roi_features, dim=1)

        scores = torch.matmul(roi_features, template_features.T)

        return scores


class ZNCCScorer(Scorer):
    """Calulate ZNCC between template(1, C) and roi(N, C) features."""

    def check_shape(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> None:
        assert template_features.dim() == 2
        assert roi_features.dim() == 2
        assert template_features.size(1) == roi_features.size(1)
        assert len(template_features) == 1

    def score(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        if len(template_features.shape) == 1:
            template_features = template_features.unsqueeze(0)
        self.check_shape(template_features, roi_features)

        template_features -= torch.mean(template_features, dim=1, keepdim=True)
        roi_features -= torch.mean(roi_features, dim=1, keepdim=True)

        template_features = F.normalize(template_features, dim=1)
        roi_features = F.normalize(roi_features, dim=1)

        scores = torch.matmul(roi_features, template_features.T)

        return scores


class MaxCosineSimilarityScorer(Scorer):
    """Calulate cosine similarity between template(M, C) and roi(N, C) features."""

    def check_shape(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> None:
        assert template_features.dim() == 2
        assert roi_features.dim() == 2
        assert template_features.size(1) == roi_features.size(1)

    def score(self, template_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        if len(template_features.shape) == 1:
            template_features = template_features.unsqueeze(0)
        self.check_shape(template_features, roi_features)

        template_features = F.normalize(template_features, dim=1)
        roi_features = F.normalize(roi_features, dim=1)

        scores = torch.matmul(roi_features, template_features.T).max(dim=1).values

        return scores
