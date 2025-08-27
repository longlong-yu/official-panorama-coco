"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
import torch


def torch_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    