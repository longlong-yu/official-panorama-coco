"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-27
@desc:   
"""
from enum import Enum


class SampleItemType(str, Enum):
    """ Sample item types """
    HAETMAP = 'heatmap'
    HBOX = 'hbox'
    SBOX = 'sbox'
    RBOX = 'rbox'
    
    @classmethod
    def of(cls, value: str):
        for t in cls:
            if t.value == value:
                return t
        return None
