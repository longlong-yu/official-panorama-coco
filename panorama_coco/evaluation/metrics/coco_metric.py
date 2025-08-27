"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-25
@desc:   
"""
from collections import OrderedDict
import datetime
import itertools
import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation import CocoMetric as MM_CocoMetric
from mmdet.structures import DetDataSample
from mmengine.fileio import dump, get_local_path
from mmengine.logging import MMLogger
from terminaltables import AsciiTable
import numpy as np
from panorama_coco.registry import METRICS, TASK_UTILS
from panorama_coco.utils.bbox import bbox_area


@METRICS.register_module()
class CocoMetric(MM_CocoMetric):
    """
    COCO evaluation metric.
    """
    def __init__(
        self,
        *,
        ann_file: Optional[str] = None,
        backend_args: dict = None,
        sort_categories: bool = False,
        coco_eval_config: dict = dict(type='COCOEval'), 
        **kwargs
    ) -> None:
        super().__init__(
            ann_file=None,
            backend_args=backend_args,
            sort_categories=False,
            **kwargs
        )
        self.ann_file = ann_file
        self.coco_eval_config = coco_eval_config
    
    @staticmethod
    def xyxy2xywh(bbox: List) -> List:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """
        return [
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ]
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            data_sample = self.prepare_sample(data_sample)
            
            # parse pred
            pred = dict()
            pred['width'] = data_sample['ori_shape'][1] 
            pred['height'] = data_sample['ori_shape'][0]
            pred['img_id'] = data_sample['img_id']
            bboxes = data_sample['pred_instances']['bboxes'].tolist()
            scores = data_sample['pred_instances']['scores'].tolist()
            labels = data_sample['pred_instances']['labels'].tolist()
            pred['anns'] = [{
                'ignore_flag': 0,
                'bbox': bboxes[i],
                'label': labels[i],
                'score': scores[i]
                # 'mask': None
            } for i in range(len(bboxes))]
            
            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1] 
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self.ann_file is None:
                gt['anns'] = []
                bboxes = data_sample['gt_instances']['bboxes'].tolist()
                labels = data_sample['gt_instances']['labels'].tolist()
                gt['anns'] = [{
                    'ignore_flag': 0,
                    'bbox': bboxes[i],
                    'label': labels[i],
                    # 'mask': None
                } for i in range(len(bboxes))]
            # add converted result to the results list
            self.results.append((self.process_hook(gt=gt, pred=pred, sample=data_sample)))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)
        gts, preds = self.prepare(gts=gts, preds=preds)
        coco_info = self._coco_info(gts)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            if self.outfile_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                outfile_prefix = os.path.join(tmp_dir.name, 'results')
            else:
                outfile_prefix = self.outfile_prefix

            if self.ann_file is None:
                # use converted gt json file to initialize coco api
                logger.info('Converting ground truth to coco format...')
                annotations = self._annotations(gts)
                coco_info['annotations'] = annotations
                coco_json_path = self._to_coco_json(
                    coco_info=coco_info, 
                    filename=f'{outfile_prefix}.gt.json'
                )
                gt_coco = COCO(coco_json_path)
            else:
                with get_local_path(
                    self.ann_file, 
                    backend_args=self.backend_args
                ) as local_path:
                    gt_coco = COCO(local_path)

            # convert predictions to coco format and dump to json file
            annotations = self._annotations(preds)
            coco_info['annotations'] = annotations
            coco_json_path = self._to_coco_json(
                coco_info=coco_info, 
                filename=f'{outfile_prefix}.bbox.json'
            )
            pred_coco = COCO(coco_json_path)
            
        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'results are saved in {os.path.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # evaluate proposal, bbox and segm
            config = self.coco_eval_config.copy()
            config['cocoGt'] = gt_coco
            config['cocoDt'] = pred_coco
            config['iouType'] = metric
            coco_eval = TASK_UTILS.build(config)
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(f'metric item "{metric_item}" is not supported')

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            if self.classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = coco_eval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(coco_eval.params.catIds) == precisions.shape[2]

                results_per_category = []
                for idx, cat_id in enumerate(coco_eval.params.catIds):
                    cat_id = cat_id.item()
                    t = []
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = gt_coco.loadCats(cat_id)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    t.append(f'{nm["name"]}')
                    t.append(f'{round(ap, 3)}')
                    eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                    # indexes of IoU  @50 and @75
                    for iou in [0, 5]:
                        precision = precisions[iou, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{round(ap, 3)}')

                    # indexes of area of small, median and large
                    for area in [1, 2, 3]:
                        precision = precisions[:, :, idx, area, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{round(ap, 3)}')
                    results_per_category.append(tuple(t))

                num_columns = len(results_per_category[0])
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [
                    'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                    'mAP_m', 'mAP_l'
                ]
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers] + [result for result in results_2d]
                table = AsciiTable(table_data)
                logger.info('\n' + table.table)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = coco_eval.stats[coco_metric_names[metric_item]]
                eval_results[key] = float(f'{round(val, 3)}')

            ap = coco_eval.stats[:6]
            logger.info(
                f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}'
            )
        return eval_results

    @staticmethod
    def prepare_sample(sample: DetDataSample):
        if 'scale_factor' in sample:
            bboxes = sample['gt_instances']['bboxes'] 
            bboxes[..., :4] /= bboxes.new_tensor(sample['scale_factor']).repeat((1, 2))
        return sample
    
    def process_hook(self, gt: dict, pred: dict, sample: DetDataSample):
        return gt, pred
    
    def prepare(self, gts: List[dict], preds: List[dict]) -> Tuple[List[dict], List[dict]]:
        for gt in gts:
            for ann in gt['anns']:
                ann['bbox'] = self.xyxy2xywh(ann['bbox'])
                ann['area'] = bbox_area(ann['bbox'])
        for pred in preds:
            for ann in pred['anns']:
                ann['bbox'] = self.xyxy2xywh(ann['bbox'])
                ann['area'] = bbox_area(ann['bbox'])
                
        return gts, preds
        
    def _coco_info(self, gts: List[dict]) -> dict:
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        
        image_infos = []
        for idx, gt in enumerate(gts):
            img_id = gt.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt['width'],
                height=gt['height'],
                file_name=''
            )
            image_infos.append(image_info)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description=f'Coco json file converted by {self.__class__}.'
        )
        return dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )

    @staticmethod
    def _annotations(gts_or_preds: List[dict]) -> List[dict]:
        annotations = []
        for idx, item in enumerate(gts_or_preds):
            img_id = item.get('img_id', idx)
            for ann in item['anns']:
                annotation = {
                    'id': len(annotations) + 1,  # coco api requires id starts with 1
                    'image_id': img_id,
                    'bbox': ann['bbox'],
                    'iscrowd': ann.get('ignore_flag', 0),
                    'category_id': int(ann['label']),
                    'area': ann['area'],
                    'score': ann.get('score', 1),
                }
                annotations.append(annotation)
        return annotations

    @staticmethod
    def _to_coco_json(coco_info: dict, filename: str) -> str:
        dump(coco_info, filename)
        return filename
