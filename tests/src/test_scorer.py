

import torch

from scorer import CosineSimilarityScorer


def test_cosine_similarity_scorer():
    scorer = CosineSimilarityScorer()
    template_features = torch.tensor([[1, 2, 3]]).float()
    roi_features = torch.tensor([[1, 2, 3], [-1, -2, -3], [1, 2, 3], [2, 4, 6]]).float()
    print(template_features.shape)
    print(roi_features.shape)
    scores = scorer.score(template_features, roi_features)
    print(scores)
    assert torch.allclose(scores, torch.tensor([1., -1., 1., 1.]))