
import torch
from mmcv.ops import batched_nms

from panorama_coco.utils.bbox import Sph2PlanarBoxTransform


class PlanarNMS:
    def __init__(self, box_formator='sph2pix'):
        self.box_formator = Sph2PlanarBoxTransform(box_formator)

    def __call__(self, boxes, scores, idxs, nms_cfg, class_agnostic=True):
        boxes_ = self.box_formator(boxes, box_version=boxes.size(1))
        det_bboxes_, keep_idxs = batched_nms(boxes_, scores, idxs, nms_cfg, class_agnostic)
        det_bboxes = boxes[keep_idxs]
        det_scores = det_bboxes_[:, -1].unsqueeze(1)
        det_bboxes = torch.cat([det_bboxes, det_scores], dim=-1)
        return det_bboxes, keep_idxs