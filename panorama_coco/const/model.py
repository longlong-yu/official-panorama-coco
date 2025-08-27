"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-27
@desc:   
"""
from enum import Enum


class ModeType(str, Enum):
    """ Mode types """
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    
    @classmethod
    def of(cls, value: str):
        for t in cls:
            if t.value == value:
                return t
        return None
