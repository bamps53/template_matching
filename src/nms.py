from detectron2.layers.rotated_boxes import pairwise_iou_rotated
import torch

from candidates import Candidates


class NMS:
    def __init__(self, score_threshold: float, iou_threshold: float):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, rois: Candidates, scores: torch.Tensor) -> Candidates:
        assert rois.rois.shape[0] == scores.shape[0]
        assert rois.rois.shape[1] == 6
        assert scores.shape[1] == 1
        assert scores.shape[0] == rois.rois.shape[0]

        rois = rois.rois[:, 1:].clone()
        scores = scores.reshape(-1)
        print("num rois: ", len(rois))

        # sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        rois = rois[sorted_indices]
        scores = scores[sorted_indices]

        # filter by score
        indices = scores > self.score_threshold
        rois = rois[indices]
        scores = scores[indices]
        print("num rois after filtered by score: ", len(rois))

        # filter by iou
        keep = []
        for i in range(len(rois)):
            if len(keep) == 0:
                keep.append(i)
                continue
            iou = pairwise_iou_rotated(rois[i:i + 1], rois[keep])
            if iou.max() < self.iou_threshold:
                keep.append(i)

        indices = torch.tensor(keep)

        rois = rois[indices]
        scores = scores[indices]
        print("num rois after filtered by iou: ", len(rois))

        return rois, scores
