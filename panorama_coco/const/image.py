"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2023-12-26
@desc:   Constant assocaited with images. 
"""
from enum import Enum


class ImageType(str, Enum):
    """ Image types """
    PLANAR = 'planar'
    SPHERE = 'sphere'
    ERP = 'erp'
