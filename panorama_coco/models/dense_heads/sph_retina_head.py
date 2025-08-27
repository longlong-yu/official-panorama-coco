"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-04
@desc:  Refer to https://github.com/AntXinyuan/sph2pob for the original implementation.
"""
import logging
from mmdet.models.dense_heads import RetinaHead
from panorama_coco.models.dense_heads.base_anchor_head import AnchorHead
from panorama_coco.registry import MODELS


logger = logging.getLogger(__name__)


@MODELS.register_module()
class SphRetinaHead(AnchorHead, RetinaHead):
    pass
