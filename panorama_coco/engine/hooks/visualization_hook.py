"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-27
@desc: 
"""
from typing import Dict, Optional, Sequence, Set, Tuple, Union
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.visualization import Visualizer
from panorama_coco.const.model import ModeType
from panorama_coco.registry import HOOKS
from pydantic import BaseModel


class VisConfig(BaseModel):
    # origin image
    image: bool = False
    
    # pred config
    # max pred bbox number in one image
    max_bbox_n: int = 0
    score_thr: float = 0.3

    # heatmap
    topk: int = 20
    alpha: float = 0.5
    arrangement: Tuple[int, int] = (4, 5)
    channel_reduction: str = 'squeeze_mean' # 'squeeze_mean' or 'select_max'
    local_maximum_kernel: int = 3 
    
    valid_names: Set[str] = set()

class ModeConfig(BaseModel):
    interval: int = 50 
    # max saved iteration number
    max_n: int = 0
    # per iteration saved number
    per_n: int = 0 
    enabled: bool = False
    counter: int = 0
    mode: ModeType = None
    channel_configs: Dict[str, VisConfig] = {}
    

@HOOKS.register_module()
class VisualizationHook(Hook):
    def __init__(
        self,
        *,
        wait_time: float = 0.,
        out_dir: Optional[str] = None,
        backend_args: dict = None,
        **kwargs
    ) -> None:
        super().__init__()
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.wait_time = wait_time
        self.out_dir = out_dir
        self.backend_args = backend_args
        
        self.mode_configs = {}
        for mode in ModeType:
            self.mode_configs[mode] = ModeConfig.model_validate(kwargs.get(mode.value, {}))
            self.mode_configs[mode].mode = mode
        
    def _check(
        self, 
        iter_index: int,
        mode: ModeType,
    ) -> bool:
        """ Check wether to draw. """
        if mode is None:
            return False
        mode_config = self.mode_configs.get(mode)
        if mode_config is None or not mode_config.enabled:
            return False
            
        if mode_config.max_n > 0 and mode_config.counter >= mode_config.max_n:
            return False
        if iter_index % mode_config.interval:
            return False
        
        return True
         
    def _before_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        mode: str = 'train'
    ) -> None:
        """
        All subclasses should override this method, if they need any
        operations before each iter.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        mode = ModeType.of(mode)
        if self._check(
            iter_index=batch_idx, 
            mode=mode
        ):
            self._visualizer.start(mode_config=self.mode_configs[mode])
        
    def _after_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[Sequence, dict]] = None,
        mode: str = 'train'
    ) -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict or Sequence, optional): Outputs from model.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        mode = ModeType.of(mode)
        if self._check(
            iter_index=batch_idx,
            mode=mode,
        ):  
            if mode == ModeType.TRAIN:
                samples=data_batch['data_samples']
            else:
                samples = outputs
             
            self._visualizer.add_channel_images(
                image_ids=samples,
                images=data_batch['inputs'],
                is_bgr=True,
            )    
            
            self._visualizer.close()
            self.mode_configs[mode].counter += 1
