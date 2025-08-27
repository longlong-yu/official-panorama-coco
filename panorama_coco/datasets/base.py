"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-23
@desc:   
"""
from typing import Any, List, Tuple, Union
from mmengine.dataset import BaseDataset as MM_BaseDataset

from panorama_coco.structures.bbox.sphere_bboxes import SphereBoxes


class BaseDataset(MM_BaseDataset):
    """
    Note:
    1. 'metainfo' should contain key 'classes'. The label id is according to the order in 'classes'.
    2. 'data_prefix' support 'img' and 'img_path' as the prefix of image path, but 'img_path' is recommended.
    3. filter_hook() is used to filter data instead of filter_data(), and self.filter_config can be used here.
    
    Args:
        limit_n: The maximum number to obtain.
    """
    def __init__(
        self, 
        *, 
        limit_n: int = 0, 
        **kwargs
    ) -> None:
        self.limit_n = limit_n
        self.category_to_label = {}
        super().__init__(**kwargs)
        
    def filter_data(self) -> List[dict]:
        if self.limit_n > 0:
            return super().filter_data()[:self.limit_n]
        else:
            return super().filter_data() 
    
    def load_data_list(self) -> List[dict]:
        """
        Returns:
            list[dict]: A list of annotation.
        """
        annotations, classes = self.load_annotations()
        self._metainfo['classes'] = classes
        self.category_to_label = {item: i for i, item in enumerate(self._metainfo['classes'])}
        
        data_list = []
        for item in self.load_raw_data(annotations):
            item = self.parse_data_info(item)
            if self.filter_hook(item):
                data_list.append(item)
            
        if self.limit_n > 0:
            return data_list[:self.limit_n]
        else:
            return data_list
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """
        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        return self.parse_raw_data(raw_data_info)
    
    def load_annotations(self) -> Tuple[Any, List[str]]:
        raise NotImplementedError(
            f'{type(self)} must implement "load_annotations" method.'
        )
     
    def load_raw_data(self, annotations: Any):
        raise NotImplementedError(
            f'{type(self)} must implement "load_raw_data" method.'
        )

    def parse_raw_data(self, raw_data_info: dict) -> dict:
        return raw_data_info
    
    # hooks
    def filter_hook(self, data_info: dict) -> bool:
        return True
    
    def load_annotations_hook(self, annotations: Any) -> None:
        pass
