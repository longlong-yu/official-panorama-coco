"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2023-12-20
@desc:   
"""
from .iou_loss import TangentIouLoss
from .sph_obb_iou_loss import Sph2ObbTransfrom

__all__ = [
    'TangentIouLoss',
    'Sph2ObbTransfrom'
]
