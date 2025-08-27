"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-24
@desc:   
"""
from enum import Enum


class ExtendType(str, Enum):
    """ Extend types """
    REFLECT = 'reflect'
    ZERO = 'zero'
    CIRCULAR = 'circular'
    REPLICATE = 'replicate'

    @classmethod
    def of(cls, value: str):
        for t in cls:
            if t.value == value:
                return t
        return None
