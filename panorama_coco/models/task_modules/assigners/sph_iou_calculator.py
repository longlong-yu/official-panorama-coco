"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-04
@desc:   refer: Sph2Pob
"""
from panorama_coco.registry import TASK_UTILS
import torch


from panorama_coco.sdk.sph2pob.bbox.iou import sph2pob_efficient_iou, naive_iou


@TASK_UTILS.register_module()
class SphOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, backend='xinyuan', box_version=4):
        self.backend = backend
        self.box_version = box_version

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 4) in
                <theta, phi, alpha, beta, (angle)> format, or shape (m, 5) in
                 <theta, phi, alpha, beta, (angle), score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 4) in
                <theta, phi, alpha, beta, (angle)> format, shape (m, 5) in
                 <theta, phi, alpha, beta, (angle), score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        #print('SphOverlaps2D.__call__()...')
        assert bboxes1.size(-1) in [0, 4, 5, 6]
        assert bboxes2.size(-1) in [0, 4, 5, 6]

        if not torch.is_tensor(bboxes1):
            bboxes1 = bboxes1.tensor
        if not torch.is_tensor(bboxes2):
            bboxes2 = bboxes2.tensor
        bboxes1 = bboxes1[..., :self.box_version]
        bboxes2 = bboxes2[..., :self.box_version]
        if self.box_version == 5:
            assert self.backend in ['xinyuan', 'planar']
        with torch.no_grad():
            overlaps = sph_overlaps(bboxes1, bboxes2, mode, is_aligned, self.backend)
        return overlaps

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

def sph_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, backend='chenbin'):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (m, 4) in <theta, phi, alpha, beta, (angle)> format
            or empty.
        bboxes2 (torch.Tensor): shape (n, 4) in <theta, phi, alpha, beta, (angle)> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    assert backend in ['xinyuan', 'planar']
    # Either the boxes are empty or the length of boxes's last dimension is 4
    #assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    #assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if backend == 'xinyuan':
        iou_calculator = sph2pob_efficient_iou
    elif backend == 'planar':
        iou_calculator = naive_iou
    else:
        raise NotImplemented('Not supported iou_calculator.')
    
    # !Note: Sph2pob uses a spherical coordinate system with longitude in [0, 360] and latitude in [0, 180], 
    # where the top-left corner of the image is the origin.
    bboxes1[..., 0] = bboxes1[..., 0] + torch.pi
    bboxes1[..., 1] = torch.pi / 2 - bboxes1[..., 1]
    bboxes2[..., 0] = bboxes2[..., 0] + torch.pi
    bboxes2[..., 1] = torch.pi / 2 - bboxes2[..., 1] 
    overlaps = iou_calculator(bboxes1, bboxes2, mode, is_aligned)
    bboxes1[..., 0] = bboxes1[..., 0] - torch.pi
    bboxes1[..., 1] = torch.pi / 2 - bboxes1[..., 1]
    bboxes2[..., 0] = bboxes2[..., 0] - torch.pi
    bboxes2[..., 1] = torch.pi / 2 - bboxes2[..., 1]
    
    return overlaps
