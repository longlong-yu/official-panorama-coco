"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-27
@desc:   
"""
import os
from typing import Any, Dict, List, Tuple, Union
import matplotlib.colors as mcolors
import mmcv
from mmdet.structures import DetDataSample
from mmengine.dist import master_only, is_main_process
from mmengine.visualization import Visualizer
from mmrotate.structures.bbox import RotatedBoxes
import numpy as np
from pydantic import BaseModel
from torch import Tensor
import torch
from panorama_coco.engine.hooks.visualization_hook import ModeConfig, VisConfig
from panorama_coco.registry import VISUALIZERS
from panorama_coco.utils.spherical import SphereImageUtils
from panorama_coco.visualization.const import SampleItemType
from panorama_coco.visualization.sphbb2erpbb import Sphbb2Erpbb


class VisChannelItem(BaseModel):
    value: Any
    item_type: SampleItemType
    
    # gt_hbox, gt_sbox, pred_hbox, pred_sbox
    labels: Tensor = None
    scores: Tensor = None

    class Config:
        arbitrary_types_allowed = True

class VisChannel(BaseModel):
    name: str
    mean: Tuple[float, float, float] = (0, 0, 0)
    std: Tuple[float, float, float] = (1, 1, 1)
    is_bgr: bool = True
    
    image_ids: List[str] = []
    samples: List[DetDataSample] = []
    images: List[Union[np.ndarray, Tensor]] = []
    channel_items: Dict[str, List[VisChannelItem]] = {}
    
    class Config:
        arbitrary_types_allowed = True

@VISUALIZERS.register_module()
class BaseVisualizer(Visualizer):
    """ 
    Add some new hooks in contrast with Visualizer.
    """
    MAIN_CHANNEL= 'main'
    
    def __init__(
        self,
        name='visualizer',
        *,
        line_width: Union[int, float] = 3,
        alpha: float = 0.8,
        **kwargs, 
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.mode_config = None
        self.channels: dict[str, VisChannel] = {}
        
        self.line_width = line_width
        self.alpha = alpha
    
    @master_only             
    def start(self, mode_config: ModeConfig):
        self.mode_config = mode_config
    
    @master_only
    def save(self):
        if self.mode_config is None or not self.mode_config.enabled:
            return    

        classes = self.dataset_meta.get('classes', ())
        palette = self._get_palette(num=len(classes))
               
        for channel_name, channel in self.channels.items():
            if not self.check_channel(channel_name):
                continue
            
            channel_config = self.mode_config.channel_configs[channel_name]
            self.prepare_sample_channel(
                channel=channel, 
                channel_config=channel_config
            )
            
            channel_len = len(channel.image_ids)
            if channel_len == 0:
                continue
            
            for idx in range(channel_len):
                if self.mode_config.per_n > 0 and idx >= self.mode_config.per_n:
                    break
                
                image_id = channel.image_ids[idx]
                prefix = f'{self.mode_config.mode.value}_{image_id}_{channel_name}'
                # draw image
                if channel.images:
                    image = channel.images[idx]
                    image = image.permute(1, 2, 0).cpu().numpy()
                    if channel.is_bgr:
                        image = mmcv.image.bgr2rgb(image)
                    for i in range(len(channel.std)):
                        image[:,:,i] = image[:,:,i] * channel.std[i] + channel.mean[i]
                    image = image.clip(0, 255).astype(np.uint8)
                    if channel_config.image:
                        self.add_image(
                            name=prefix,
                            image=image,
                            step=self.mode_config.counter,
                        )
                else:
                    image = None
                    
                for name, channel_items in channel.channel_items.items():
                    if name not in channel_config.valid_names:
                        continue
                    
                    channel_item = channel_items[idx]
                    scores = channel_item.scores
                    value = channel_item.value.cpu()
                    labels = channel_item.labels
                    save_name = f'{prefix}_{name}'
                    
                    # score_thr
                    if scores is not None and channel_config.score_thr > 0:
                        mask = scores > channel_config.score_thr
                        scores = scores[mask]
                        if labels is not None:
                            labels = labels[mask]
                        value = value[mask, ...]
                    
                    # max_bbox_n
                    if channel_config.max_bbox_n > 0:
                        if scores is not None:
                            scores = scores[:channel_config.max_bbox_n]
                        if labels is not None:
                            labels = labels[:channel_config.max_bbox_n]
                        value = value[:channel_config.max_bbox_n, ...]
                    
                    if scores is not None:
                        scores = scores.tolist()
                    if labels is not None:
                        labels = labels.tolist()    
                        colors = [palette[item] for item in labels]
                    else:
                        # colors = [palette[30]] * value.shape[0]
                        colors = [(154, 205, 50)] * value.shape[0]
                    
                    new_image = None
                    if channel_item.item_type == SampleItemType.HBOX:  
                        self.set_image(image)
                        self.draw_bboxes(
                            value,
                            edge_colors=colors,
                            alpha=self.alpha,
                            line_widths=self.line_width,
                        )
                        self.draw_labels(
                            positions=value[:, :2],
                            areas=(value[:, 3] - value[:, 1]) * (value[:, 2] - value[:, 0]),
                            labels=labels,
                            classes=classes,
                            colors=colors,
                            scores=scores,
                        )
                        new_image = self.get_image()
                    elif channel_item.item_type == SampleItemType.RBOX:
                        self.set_image(image)
                        positions = self.draw_rboxes(
                            bboxes=value,
                            colors=colors
                        )
                        self.draw_labels(
                            positions=positions,
                            areas=value[:, 2] * value[:, 3],
                            labels=labels,
                            classes=classes,
                            colors=colors,
                            scores=scores,
                        )
                        new_image = self.get_image() 
                    elif channel_item.item_type == SampleItemType.SBOX:
                        self.set_image(image)
                        positions = self.draw_sboxes(
                            bboxes=value,
                            erp_w=image.shape[1],
                            erp_h=image.shape[0],
                            colors=colors
                        )
                        self.draw_labels(
                            positions=positions,
                            areas=value[:, 2] * value[:, 3] * (image.shape[0] * image.shape[0] / 32400),
                            labels=labels,
                            classes=classes,
                            colors=colors,
                            scores=scores,
                        )
                        new_image = self.get_image() 
                    elif channel_item.item_type == SampleItemType.HAETMAP:
                        new_image = self.draw_featmap(
                            featmap=value,
                            overlaid_image=image,
                            channel_reduction=channel_config.channel_reduction,
                            topk=channel_config.topk,
                            arrangement=channel_config.arrangement,
                            resize_shape=None,
                            alpha=channel_config.alpha,
                        )
                        
                    if new_image is not None:
                        self.add_image(
                            name=save_name,
                            image=new_image,
                            step=self.mode_config.counter,
                        )
             
    @staticmethod
    def _get_palette(num: int, mode: str = 'css4') -> List[Tuple]:
        palette = []
        if mode == 'ccs4':
            colors = list(mcolors.CSS4_COLORS.values())
        else:
            colors = list(mcolors.XKCD_COLORS.values())  
        for i in range(num):
            color = mcolors.to_rgb(colors[i])
            palette.append((
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            ))
        return palette   
        
    def draw_labels(
        self, 
        positions: Tensor,
        areas: Tensor,
        labels: List[int],
        classes: List[str],
        colors: List[Tuple[int, int, int]],
        scores: List[float] = None,
    ):
        positions = positions + self.line_width
        scales = self._get_adaptive_scales(areas).tolist()
        label_texts = []
        for i in range(positions.shape[0]):
            if labels:
                label_text = classes[labels[i]] if classes is not None else f'class {labels[i]}'
            else:
                label_text = ''
            if scores:
                score = round(float(scores[i]) * 100, 1)
                label_text += f': {score}'
            if not label_text:
                label_text = f'label-{i}'
            label_texts.append(label_text)

        self.draw_texts(
            label_texts,
            positions,
            colors=colors,
            font_sizes=[int(13 * item) for item in scales],
            bboxes={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            }
        )
    
    def draw_sboxes(
        self,
        bboxes: Tensor,
        erp_w: int,
        erp_h: int,
        colors: List[Tuple[int, int, int]],
    ):
        bboxes = bboxes.clone()
        bboxes[:, 2:4] = bboxes[:, 2:4] * 180 / torch.pi
        label_xs = []
        label_ys = []
        for i, bbox in enumerate(bboxes):
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            
            BFoV = Sphbb2Erpbb(
                erp_w, erp_h,
                view_angle_w=bbox[2].numpy(),
                view_angle_h=bbox[3].numpy(),
            )
            position_x, position_y = BFoV._sample_points(
                bbox.numpy(), erp_w, erp_h, border_only=True
            )
            position_x = torch.from_numpy(position_x).reshape(-1, 1)
            position_y = torch.from_numpy(position_y).reshape(-1, 1)
            centers = torch.cat([position_x, position_y], dim=-1).int()
            self.draw_circles(
                center=centers,
                radius=centers.new_ones(centers.shape[:1]) * self.line_width * 0.5,
                edge_colors=colors[i],
                alpha=self.alpha,
                line_widths=self.line_width
            )
            
            slices = [
                0,
                BFoV._ANGy.shape[1] - 1, 
                BFoV._ANGy.shape[1],
                BFoV._ANGy.shape[1] * 2 - 1,
            ]
            vertices = [[position_x[item, 0], position_y[item, 0]] for item in slices]
            ct_xyz = SphereImageUtils.theta_phi2xyz(bbox[0:2])
            vertices_xyz = SphereImageUtils.pxpy2xyz(
                ct_xyz.new_tensor(vertices).permute(1, 0),
                w=erp_w,
                h=erp_h
            )
            
            axis_left = torch.cross(ct_xyz, ct_xyz.new_tensor([0, 1, 0]))
            vertices_cos = []
            for vertice in vertices_xyz.permute(1, 0):
                vertices_cos.append((vertice, vertice @ axis_left))
            vertices_cos = sorted(vertices_cos, key=lambda e: e[1])
            if vertices_cos[0][1] * vertices_cos[-1][1] < 0:
                left_vertices = vertices_cos[-2:]
            else:
                left_vertices = vertices_cos[:2]
            if left_vertices[0][0][1] >= left_vertices[1][0][1]:
                left_vertice = left_vertices[0][0]
            else:
                left_vertice = left_vertices[1][0]
            
            left_pxpy = SphereImageUtils.xyz2pxpy(left_vertice, w=erp_w, h=erp_h)
            label_xs.append(left_pxpy[0])
            label_ys.append(left_pxpy[1])
             
        label_xs = torch.tensor(label_xs, device=bboxes.device).view(-1, 1)
        label_ys = torch.tensor(label_ys, device=bboxes.device).view(-1, 1)
        return torch.cat((label_xs, label_ys), dim=-1).int()
     
    @staticmethod
    def _get_adaptive_scales(
        areas: Tensor,
        min_area: int = 800,
        max_area: int = 30000
    ) -> np.ndarray:
        """Get adaptive scales according to areas.

        The scale range is [0.5, 1.0]. When the area is less than
        ``min_area``, the scale is 0.5 while the area is larger than
        ``max_area``, the scale is 1.0.

        Args:
            areas (ndarray): The areas of bboxes or masks with the
                shape of (n, ).
            min_area (int): Lower bound areas for adaptive scales.
                Defaults to 800.
            max_area (int): Upper bound areas for adaptive scales.
                Defaults to 30000.

        Returns:
            ndarray: The adaotive scales with the shape of (n, ).
        """
        scales = 0.5 + (areas - min_area) // (max_area - min_area)
        return torch.clamp(scales, max=1.0, min=0.5)
    
    def draw_rboxes(self, bboxes: Tensor, colors: List[Tuple[int, int, int]]):
        """
        refer: https://github.com/open-mmlab/mmrotate/blob/dev-1.x/mmrotate/visualization/local_visualizer.py
        """
        polygons = RotatedBoxes(bboxes).convert_to('qbox').tensor
        polygons = polygons.reshape(-1, 4, 2)
        polygons = [p for p in polygons]
        self.draw_polygons(
            polygons,
            edge_colors=colors,
            alpha=self.alpha,
            line_widths=self.line_width
        )

        return bboxes[..., 0:2] + self.line_width
    
    @master_only 
    def close(self):
        self.save()
        self.channels = {}
        self.mode_config = None
        super().close()
    
    def check_mode(self) -> bool:
        return is_main_process() and self.mode_config and self.mode_config.enabled
      
    ############################# channel ############################################
    def check_channel(self, channel: str) -> bool:
        if not self.check_mode():
            return False
        
        channel_config = self.mode_config.channel_configs.get(channel)
        if not channel_config:
            return False
        
        return True
    
    def add_channel_image(
        self,
        *,
        channel: str = MAIN_CHANNEL,
        image_id: Union[str, DetDataSample],
        image: np.ndarray = None,
        mean: Tuple[float, float, float] = (0, 0, 0),
        std: Tuple[float, float, float] = (1, 1, 1),
        is_bgr: bool = True
    ):
        return self.add_channel_images(
            channel=channel,
            image_ids=[image_id],
            images=[image],
            mean=mean,
            std=std,
            is_bgr=is_bgr
        )
    
    def add_channel_images(
        self,
        *,
        channel: str = MAIN_CHANNEL,
        image_ids: List[Union[str, DetDataSample]],
        images: List[Union[np.ndarray, Tensor]] = [],
        mean: Tuple[float, float, float] = (0, 0, 0),
        std: Tuple[float, float, float] = (1, 1, 1),
        is_bgr: bool = True
    ):  
        if not self.check_channel(channel):
            return self
              
        if channel not in self.channels:
            self.channels[channel] = VisChannel(name=channel)

        if len(self.channels[channel].images) == 0:
            self.channels[channel].mean = mean
            self.channels[channel].std = std
            self.channels[channel].is_bgr = is_bgr
            
        self.channels[channel].images += images
        if isinstance(image_ids[0], str):
            self.channels[channel].image_ids += image_ids
        else:
            self.channels[channel].samples += image_ids
            
        return self

    def add_channel_item(
        self,
        *,                
        channel: str = MAIN_CHANNEL,
        name: str, 
        value: Any, 
        item_type: SampleItemType,
        **kwargs
    ):
        return self.add_channel_items(
            channel=channel,
            name=name,
            values=[value],
            item_type=item_type,
            **kwargs
        )

    def add_channel_items(
        self,
        *,                
        channel: str = MAIN_CHANNEL,
        name: str, 
        values: List[Any], 
        item_type: SampleItemType,
        **kwargs
    ):
        if not self.check_channel(channel):
            return self
        
        if channel not in self.channels:
            self.channels[channel] = VisChannel(name=channel)
        if name not in self.channels[channel].channel_items:
            self.channels[channel].channel_items[name] = []
        
        for value in values:
            channel_item = VisChannelItem(
                value=value,
                item_type=item_type,
                **kwargs,
            )
            self.channels[channel].channel_items[name].append(channel_item)
        return self

    def prepare_sample_channel(self, channel: VisChannel, channel_config: VisConfig): 
        if len(channel.samples) == 0:
                return

        channel.image_ids = []
        for sample in channel.samples:
            sample = sample.cpu()
            channel.image_ids.append(os.path.basename(sample.img_path).split('.')[0])
            
            # gt
            bboxes = sample.gt_instances.bboxes 
            if not torch.is_tensor(bboxes):
                bboxes = bboxes.tensor
            labels = sample.gt_instances.labels
            if 'gt_hbox' in channel_config.valid_names:
                self.add_channel_item(
                    channel=channel.name,
                    name='gt_hbox',
                    value=bboxes,
                    item_type=SampleItemType.HBOX,
                    labels=labels
                )
            if 'gt_rbox' in channel_config.valid_names:
                self.add_channel_item(
                    channel=channel.name,
                    name='gt_rbox',
                    value=bboxes,
                    item_type=SampleItemType.RBOX,
                    labels=labels
                )
            if 'gt_sbox' in channel_config.valid_names:
                self.add_channel_item(
                    channel=channel.name,
                    name='gt_sbox',
                    value=bboxes,
                    item_type=SampleItemType.SBOX,
                    labels=labels
                )
            
            # pred
            if 'pred_instances' in sample:
                preds = sample.pred_instances.bboxes
                if not torch.is_tensor(preds):
                    preds = preds.tensor
                pred_labels = sample.pred_instances.labels
                scores = sample.pred_instances.scores
                
                if 'pred_hbox' in channel_config.valid_names:
                    self.add_channel_item(
                        channel=channel.name,
                        name='pred_hbox',
                        value=preds,
                        item_type=SampleItemType.HBOX,
                        labels=pred_labels,
                        scores=scores
                    )
                if 'pred_rbox' in channel_config.valid_names:
                    self.add_channel_item(
                        channel=channel.name,
                        name='pred_rbox',
                        value=preds,
                        item_type=SampleItemType.RBOX,
                        labels=pred_labels,
                        scores=scores
                    )
                if 'pred_sbox' in channel_config.valid_names:
                    self.add_channel_item(
                        channel=channel.name,
                        name='pred_sbox',
                        value=preds,
                        item_type=SampleItemType.SBOX,
                        labels=pred_labels,
                        scores=scores
                    )
    