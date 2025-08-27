"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-28
@desc:   
"""
from torch import Tensor


def normalize(x: Tensor, *, max_, min_=0, eps=0) -> Tensor:
    if not x.numel():
        return x
    
    x = (x - min_) % (max_ - min_) + min_
    return x.clamp_(min=min_ + eps, max=max_ - eps)
