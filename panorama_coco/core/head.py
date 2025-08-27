"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-18
@desc:   
"""

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple
from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

class BaseHead(BaseModule, metaclass=ABCMeta):
    """
    Base class for Head in the same fashion as mmdet.BaseDenseHead, but with 
    more clear hook methods and more generic method declaration to be inherited.
    Some methods are removed as they're not commonly used or not very well realised.

    1. The ``init_weights`` method is used to initialize head's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of head,
    which includes two steps: (1) the head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    4. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call head's ``forward``,
    ``loss_by_feat`` and ``predict_by_feat`` methods in order.  If one-stage is
    used as RPN, the head needs to return both losses and predictions.
    This predictions is used as the proposal of roihead.

    .. code:: text

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(
        self, 
        init_cfg: OptMultiConfig = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self) -> None:
        """ Initialize the weights. """
        super().init_weights()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor]]:
        """
        Forward features.

        Args:
            x (tuple[Tensor]): Features from the upstream network.

        Returns: Predict results.
        """
        return multi_apply(self.forward_single, x)

    @abstractmethod
    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Forward feature.

        Args:
            x (Tensor): Feature.

        Returns: Predict result.
        """
        pass
    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList, **kwargs) -> dict:
        """
        Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        y = self(x)
        ground_truths = self.get_targets(batch_data_samples, feats=y, **kwargs)
        return self.loss_by_feat(*y, *ground_truths, **kwargs)

    @abstractmethod
    def loss_by_feat(self, *features_and_gts, **kwargs) -> dict:
        """
        Calculate the loss based on the features extracted by the detection
        head.
        """
        pass
   
    @abstractmethod
    def get_targets(
        self, 
        batch_data_samples: SampleList, 
        *, 
        feats: Tuple[Tensor], 
        **kwargs
    ) -> Tuple[Any, ...]:
        """
        Compute targets in multiple images.

        Args:
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Any, ...]: Targets
        """
        pass
    
    def predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        **kwargs,
    ) -> InstanceList:
        """
        Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        y = self(x)

        return self.predict_by_feat(
            *y, batch_img_metas=batch_img_metas, **kwargs
        )

    @abstractmethod
    def predict_by_feat(
        self,
        *features: Tuple[List[Tensor]],
        batch_img_metas: Optional[List[dict]] = None,
        **kwargs,
    ) -> InstanceList:
        """
        Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            features (Tuple[list[Tensor]]): Features.
            bbox_preds (list[Tensor]): Box energies / deltas for all
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image. 
        """
        pass

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        **kwargs,
    ) -> Tuple[dict, InstanceList]:
        """
        Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image.
        """
        y = self(x)
        groud_truths = self.get_targets(batch_data_samples, **kwargs)
        losses = self.loss_by_feat(*y, *groud_truths, **kwargs)
        
        batch_img_metas = [item.metainfo for item in batch_data_samples]
        predictions = self.predict_by_feat(
            *y, batch_img_metas=batch_img_metas, **kwargs
        )
        
        return losses, predictions
