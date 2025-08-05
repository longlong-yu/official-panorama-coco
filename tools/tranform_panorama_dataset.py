"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-08
@desc:   
"""
import json
import math
from multiprocessing import Pool
import random
import time
from typing import List
import mmcv
import os
import sys
sys.path.append('.')
import torch
from torch import Tensor
from panorama_coco.const.transforms import ExtendType
from panorama_coco.datasets.transforms.transforms import SquareToERP
from panorama_coco.structures.bbox.sphere_bboxes import SphereBoxes
from panorama_coco.utils.spherical import ImageUtils, SphereImageUtils


def _slice_anns(
    anns: dict,
    process_n: int,
    classes=List[str],
    random_faces: bool=False
):
    """ Slice anns by classes and process_n. """
    category_ids = {item['id'] for item in anns['categories'] if item['name'] in (classes or [])}

    ann_infos_dict = {}
    for ann in anns['annotations']:
        if classes is not None and ann['category_id'] not in category_ids:
            continue
        
        if ann['image_id'] not in ann_infos_dict:
            ann_infos_dict[ann['image_id']] = {
                'image': None,
                'annotations': [], 
            }
        ann_infos_dict[ann['image_id']]['annotations'].append(ann)

    for meta in anns['images']:
        if meta['id'] in ann_infos_dict:
            ann_infos_dict[meta['id']]['image'] = meta

    ann_infos = sorted(ann_infos_dict.values(), key=lambda e: e['image']['id'])
    
    # Randomly sample 5 images.
    face_num = 5
    face_anns = []
    image_size = len(ann_infos)
    if random_faces:
        for i in range(face_num):
            anns = random.sample(ann_infos, image_size)
            face_anns.append(anns)
        for i in range(image_size):
            ann_infos[i]['face_anns'] = [face_anns[j][i].copy() for j in range(face_num)]
    
    if image_size % process_n == 0:
        process_size = image_size // process_n
    else:
        process_size = (image_size + process_n - image_size % process_n) // process_n
    
    for i in range(process_n):
        yield ann_infos[i * process_size: (i + 1) * process_size]


def _init_ann(ann: dict, rbox: Tensor, sbox: Tensor):
    """ Initialize annotation info. """
    return {
        'origin_id': ann['id'],
        'origin_bbox': rbox.tolist(),
        'iscrowd': ann['iscrowd'],
        'image_id': ann['image_id'],
        'category_id': ann['category_id'],
        'area': SphereImageUtils.sbox_area(sbox).item(),
        'segmentation': [],
    }


def _zero_points(meta: dict, points: Tensor):
    """ Compute the convex bounding polygon of the F-face during padding. """
    points = points.permute(1, 0)
    points = ImageUtils.pxpy2xy(points, w=meta['cube_edge'], h=meta['cube_edge'])
    points = SphereImageUtils.xy2theta_phi_gnomonic(points, radius=meta['cube_edge'] // 2)
    points = SphereImageUtils.theta_phi2xyz(points)

    UD_dict = {
        '_': points,
    }
    return [UD_dict]

    
def _zero_anns(ann: dict, F_hulls: List[dict], rbox: Tensor):
    """ Compute spherical annotations based on the bounding polygon of the F-face. """
    new_anns = []
    for UD_dict in F_hulls:        
        for kk, vv in UD_dict.items():
            sbox = SphereImageUtils.sbox_from_convex_hull(vv.permute(1, 0))
            new_ann = _init_ann(ann=ann, rbox=rbox, sbox=sbox)
            if new_ann['area'] == 0 or math.isnan(new_ann['area']):
                continue
            
            # F
            sbox = sbox.tolist()
            new_ann['orientation'] = f'F{kk}'
            new_ann['bbox'] = sbox
            new_anns.append(new_ann)
    return new_anns


def _reflect_points(meta: dict, points: Tensor):
    """ Compute the convex bounding polygon of the F-face during the padding process. """
    pad_dict = {
        '': points,
    }
    # Check whether the padding region contains the complete annotation box.
    # left
    pad_n = math.ceil(meta['pad'][0] / meta['origin_w'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        if pad_i % 2 == 0:
            new_tmp[:, 0] =  2 * meta['pad'][0] - pad_i * meta['origin_w'] - new_tmp[:, 0]
            new_tmp = new_tmp.flip(dims=[0])
        else:
            new_tmp[:, 0] -= (pad_i + 1) * meta['origin_w']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 0] >= 0):
            pad_dict[f'{"L" * (pad_i + 1)}'] = new_tmp
    # right
    pad_n = math.ceil(meta['pad'][1] / meta['origin_w'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        if pad_i % 2 == 0:
            new_tmp[:, 0] =  2 * meta['pad'][1] + (pad_i + 2) * meta['origin_w'] - new_tmp[:, 0]
            new_tmp = new_tmp.flip(dims=[0])
        else:
            new_tmp[:, 0] += (pad_i + 1) * meta['origin_w']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 0] < meta['cube_edge']):
            pad_dict[f'{"R" * (pad_i + 1)}'] = new_tmp
    # up
    pad_n = math.ceil(meta['pad'][2] / meta['origin_h'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        if pad_i % 2 == 0:
            new_tmp[:, 1] =  2 * meta['pad'][2] - pad_i * meta['origin_h'] - new_tmp[:, 1]
            new_tmp = new_tmp.flip(dims=[0])
        else:
            new_tmp[:, 1] -= (pad_i + 1) * meta['origin_h']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 1] >= 0):
            pad_dict[f'{"U" * (pad_i + 1)}'] = new_tmp
    # down
    pad_n = math.ceil(meta['pad'][3] / meta['origin_h'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        if pad_i % 2 == 0:
            new_tmp[:, 1] =  2 * meta['pad'][3] + (pad_i + 2) * meta['origin_h'] - new_tmp[:, 1]
            new_tmp = new_tmp.flip(dims=[0])
        else:
            new_tmp[:, 1] += (pad_i + 1) * meta['origin_h']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 1] < meta['cube_edge']):
            pad_dict[f'{"D" * (pad_i + 1)}'] = new_tmp
    
    rets = []                        
    for k, v in pad_dict.items():
        contained_u = True
        contained_d = True
        for point in v:
            # Check if it is on the U face.
            if point[0] < point[1] or point[0] + point[1] > meta['cube_edge']:
                contained_u = False
            # Check if it is on the D face.
            if point[0] > point[1] or point[0] + point[1] < meta['cube_edge']:
                contained_d = False

        points = v.permute(1, 0)
        points = ImageUtils.pxpy2xy(points, w=meta['cube_edge'], h=meta['cube_edge'])
        points = SphereImageUtils.xy2theta_phi_gnomonic(points, radius=meta['cube_edge'] // 2)
        points = SphereImageUtils.theta_phi2xyz(points)

        UD_dict = {
            f'_{k}': points,
        }
        if contained_u:
            new_tmp = points.clone()
            new_tmp[1] = points[2]
            new_tmp[2] = points[1]
            UD_dict[f'U_{k}'] = new_tmp.flip(dims=[1])
        if contained_d:
            new_tmp = points.clone()
            new_tmp[1] = -points[2]
            new_tmp[2] = -points[1]
            UD_dict[f'D_{k}'] = new_tmp.flip(dims=[1])
        
        rets.append(UD_dict)
    return rets

    
def _reflect_anns(ann: dict, F_hulls: List[dict], rbox: Tensor):
    """ Compute spherical annotations from the bounding polygon of the F-face. """
    new_anns = []
    for UD_dict in F_hulls:        
        for kk, vv in UD_dict.items():
            sbox = SphereImageUtils.sbox_from_convex_hull(vv.permute(1, 0))
            new_ann = _init_ann(ann=ann, rbox=rbox, sbox=sbox)
            if new_ann['area'] == 0 or math.isnan(new_ann['area']):
                continue
            
            # F
            sbox = sbox.tolist()
            if sbox[0] >= 0:
                b_sbox = [sbox[0] - torch.pi, sbox[1], sbox[2], sbox[3], sbox[4]]
            else:
                b_sbox = [sbox[0] + torch.pi, sbox[1], sbox[2], sbox[3], sbox[4]]
            l_sbox = [-sbox[0] - torch.pi/2, sbox[1], sbox[2], sbox[3], -sbox[4]]
            r_sbox = [-sbox[0] + torch.pi/2, sbox[1], sbox[2], sbox[3], -sbox[4]]
            sbox_dict = {
                f'F{kk}': sbox,
                f'B{kk}': b_sbox,
                f'L{kk}': l_sbox,
                f'R{kk}': r_sbox, 
            }
        
            for k,v in sbox_dict.items():
                new_ann['orientation'] = k
                new_ann['bbox'] = v
                new_anns.append(new_ann.copy())
    return new_anns


def _circular_points(meta: dict, points: Tensor):
    """ Compute the convex bounding polygon of the F-face during padding. """
    pad_dict = {
        '': points,
    }
    # Check whether the padding region contains the entire annotation box.
    # left
    pad_n = math.ceil(meta['pad'][0] / meta['origin_w'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        new_tmp[:, 0] -= (pad_i + 1) * meta['origin_w']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 0] >= 0):
            pad_dict[f'{"L" * (pad_i + 1)}'] = new_tmp
    # right
    pad_n = math.ceil(meta['pad'][1] / meta['origin_w'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        new_tmp[:, 0] += (pad_i + 1) * meta['origin_w']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 0] < meta['cube_edge']):
            pad_dict[f'{"R" * (pad_i + 1)}'] = new_tmp
    # up
    pad_n = math.ceil(meta['pad'][2] / meta['origin_h'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        new_tmp[:, 1] -= (pad_i + 1) * meta['origin_h']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 1] >= 0):
            pad_dict[f'{"U" * (pad_i + 1)}'] = new_tmp
    # down
    pad_n = math.ceil(meta['pad'][3] / meta['origin_h'])
    for pad_i in range(pad_n):
        new_tmp = points.clone()
        new_tmp[:, 1] += (pad_i + 1) * meta['origin_h']
        if pad_i != pad_n - 1 or torch.all(new_tmp[:, 1] < meta['cube_edge']):
            pad_dict[f'{"D" * (pad_i + 1)}'] = new_tmp
    
    rets = []                        
    for k, v in pad_dict.items():
        contained_u = True
        contained_d = True
        for point in v:
            # Check whether it lies on the D face.
            if point[0] < point[1] or point[0] + point[1] > meta['cube_edge']:
                contained_d = False
            # Check whether it lies on the U face.
            if point[0] > point[1] or point[0] + point[1] < meta['cube_edge']:
                contained_u = False

        points = v.permute(1, 0)
        points = ImageUtils.pxpy2xy(points, w=meta['cube_edge'], h=meta['cube_edge'])
        points = SphereImageUtils.xy2theta_phi_gnomonic(points, radius=meta['cube_edge'] // 2)
        points = SphereImageUtils.theta_phi2xyz(points)

        UD_dict = {
            f'_{k}': points,
        }
        if contained_u:
            new_tmp = points.clone()
            new_tmp[1] = points[2]
            new_tmp[2] = -points[1]
            UD_dict[f'U_{k}'] = new_tmp
        if contained_d:
            new_tmp = points.clone()
            new_tmp[1] = -points[2]
            new_tmp[2] = points[1]
            UD_dict[f'D_{k}'] = new_tmp
        
        rets.append(UD_dict)
    return rets


def _circular_anns(ann: dict, F_hulls: List[dict], rbox: Tensor):
    """ Compute spherical annotations from the convex bounding polygon of the F-face. """
    new_anns = []
    for UD_dict in F_hulls:        
        for kk, vv in UD_dict.items():
            sbox = SphereImageUtils.sbox_from_convex_hull(vv.permute(1, 0))
            new_ann = _init_ann(ann=ann, rbox=rbox, sbox=sbox)
            if new_ann['area'] == 0 or math.isnan(new_ann['area']):
                continue
            
            # F
            sbox = sbox.tolist()
            if sbox[0] >= 0:
                b_sbox = [sbox[0] - torch.pi, sbox[1], sbox[2], sbox[3], sbox[4]]
            else:
                b_sbox = [sbox[0] + torch.pi, sbox[1], sbox[2], sbox[3], sbox[4]]
            l_sbox = [sbox[0] - torch.pi/2, sbox[1], sbox[2], sbox[3], sbox[4]]
            r_sbox = [sbox[0] + torch.pi/2, sbox[1], sbox[2], sbox[3], sbox[4]]
            sbox_dict = {
                f'F{kk}': sbox,
                f'B{kk}': b_sbox,
                f'L{kk}': l_sbox,
                f'R{kk}': r_sbox, 
            }
        
            for k,v in sbox_dict.items():
                new_ann['orientation'] = k
                new_ann['bbox'] = v
                new_anns.append(new_ann.copy())
    return new_anns


def _anns_from_mask(ann: dict, meta: dict, extend_type: ExtendType):
    """ Compute spherical annotations from the mask. """
    # Annotate the F, B, L, and R faces.
    mask = torch.tensor(ann['segmentation'][0])
    rbox = ImageUtils.rbox_from_masks([mask])
    tmp = ImageUtils.convex_hull([mask], clockwise=True)
    # Some masks contain only a single point.
    if tmp.shape[0] < 3:
        return []
    
    # Adjust coordinates based on padding.
    rbox[0] += meta['pad'][0]
    rbox[1] += meta['pad'][2]
    tmp[:, 0] += meta['pad'][0]
    tmp[:, 1] += meta['pad'][2] 
    
    if extend_type == ExtendType.ZERO:
        F_hulls = _zero_points(meta=meta, points=tmp)
        new_anns = _zero_anns(ann=ann, F_hulls=F_hulls, rbox=rbox)
    elif extend_type == ExtendType.REFLECT:
        F_hulls = _reflect_points(meta=meta, points=tmp)
        new_anns = _reflect_anns(ann=ann, F_hulls=F_hulls, rbox=rbox)
    elif extend_type == ExtendType.CIRCULAR:
        F_hulls = _circular_points(meta=meta, points=tmp)
        new_anns = _circular_anns(ann=ann, F_hulls=F_hulls, rbox=rbox)
    else:
        raise Exception(f'Invalid ExtendType {extend_type} !')
    return new_anns
 

def _process_ann_info(
    process_idx: int, 
    ann_infos: List[dict], 
    config: dict,
    src_dir: str,
    dest_dir: str,
    height: int,
    width: int,
    extend_type: ExtendType,
    random_faces: bool,
    batch_size: int = 10,
):  
    progress_items = [] 
    progress_bar = f'{config["data_prefix"] + "-" + str(process_idx):<12}: ' \
        'images %d/%d, anns %d/%d | time: %ds|read image: %ds|write image: %ds|pad: %ds|transform: %ds|label: %ds'
    progress_items = [0, len(ann_infos), 0, 0, 0, 0, 0, 0, 0, 0]
    for ann_info in ann_infos:
        progress_items[3] += len(ann_info['annotations'])
    
    transformer_erp = SquareToERP(
        height=height,
        width=width,
        need_permute=True,
        extend_mode=extend_type,
    )
    
    # Load processed annotations.
    image_f_path = os.path.join(dest_dir, config['ann_file'])
    tmp_dir = os.path.dirname(image_f_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_fn = os.path.basename(image_f_path)
    image_f_path = os.path.join(tmp_dir, f'image_{process_idx}_{tmp_fn}')
    image_meta_dict = {}
    if os.path.exists(image_f_path):
        with open(image_f_path, 'r') as f:
            for line in f.readlines():
                if not line:
                    continue
                meta = json.loads(line.strip())
                # Delete saved face_anns from old files.
                if not random_faces and 'face_anns' in meta:
                    del meta['face_anns']
                image_meta_dict[meta['id']] = meta
    image_f = open(image_f_path, 'a')
    
    ann_f_path = os.path.join(dest_dir, config['ann_file'])
    tmp_dir = os.path.dirname(ann_f_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_fn = os.path.basename(ann_f_path)
    ann_f_path = os.path.join(tmp_dir, f'ann_{process_idx}_{tmp_fn}')
    image_ann_dict = {}
    if os.path.exists(ann_f_path):
        with open(ann_f_path, 'r') as f:
            for line in f.readlines():
                if not line:
                    continue
                ann = json.loads(line.strip())
                if (not random_faces and ann['orientation'] == 'F_') \
                    or (random_faces and ann['orientation'] in ('F_', 'B_', 'L_', 'R_', 'U_', 'D_')):
                    image_ann_dict[ann['origin_id']] = ann
    ann_f = open(ann_f_path, 'a')
      
    time_start = time.time()
    for ann_info in ann_infos:
        delta_time = {
            'read': 0,
            'write': 0,
            'transform': 0,
            'pad': 0,
            'label': 0,
        }
        
        # process image meta
        meta = ann_info['image']
        if meta['id'] in image_meta_dict:
            new_meta = image_meta_dict.pop(meta['id'])
        else:
            t1 = time.time()
            image_path = os.path.join(src_dir, config['data_prefix'], meta['file_name'])
            image = mmcv.imread(image_path)
            t2 = time.time()
            delta_time['read'] += t2 - t1
            t1 = t2
            
            # planar to erp requires size must be 2 times
            image, pad = ImageUtils.center_square_pad(
                input=image,
                extend_type=extend_type,
                size_divisor=2,
            )
            t2 = time.time()
            delta_time['pad'] += t2 - t1
            t1 = t2
            
            # pad random face image
            face_images = []
            face_metas = []
            if random_faces:    
                for face_ann in ann_info['face_anns']:
                    face_meta = face_ann['image'] 
                    face_image_path = os.path.join(src_dir, config['data_prefix'], face_meta['file_name'])
                    face_image = mmcv.imread(face_image_path)
                    face_image, face_pad = ImageUtils.center_square_pad(
                        input=face_image,
                        extend_type=extend_type,
                        size_divisor=2,
                    )
                    face_images.append(face_image)
                     
                    new_face_meta = {
                        'id': face_meta['id'],
                        'height': height,
                        'width': width,
                        'file_name': face_meta['file_name'],
                        'origin_h': face_meta['height'],
                        'origin_w': face_meta['width'],
                        'cube_edge': face_image.shape[0] ,
                        'pad': face_pad,
                    }
                    face_metas.append(new_face_meta)
                        
            cube_edge = image.shape[0]
            new_image_path = os.path.join(dest_dir, config['data_prefix'], meta['file_name'])
            if not os.path.exists(new_image_path):
                # Align the image based on the F face.
                for face_i, face_image in enumerate(face_images):
                    if face_metas[face_i]['cube_edge'] != cube_edge:
                        face_images[face_i] = mmcv.imresize_like(face_image, image)
                       
                image = transformer_erp({
                    'img': image,
                    'face_images': face_images,
                })['img']
                t2 = time.time()
                delta_time['transform'] += t2 - t1
                t1 = t2
                
                mmcv.imwrite(
                    img=image,
                    file_path=new_image_path,
                    auto_mkdir=True,
                )
                t2 = time.time()
                delta_time['write'] += t2 - t1
                
            new_meta = {
                'id': meta['id'],
                'height': height,
                'width': width,
                'file_name': meta['file_name'],
                'origin_h': meta['height'],
                'origin_w': meta['width'],
                'cube_edge': cube_edge,
                'pad': pad,
                'face_metas': face_metas,
                'face_anns': ann_info['face_anns'] if random_faces else [],
            }
            image_f.write(json.dumps(new_meta) + '\n')
            image_f.flush()

        t1 = time.time()
        # Compute annotations based on the ERP projection expanded from a single image.
        if not random_faces:
            for ann in ann_info['annotations']:
                progress_items[2] += 1
                if ann['iscrowd'] or ann['id'] in image_ann_dict:
                    continue
                
                new_anns = _anns_from_mask(ann=ann, meta=new_meta, extend_type=extend_type)            
                lines = []
                for item in new_anns:
                    lines.append(json.dumps(item) + '\n')
                ann_f.writelines(lines)
                ann_f.flush()
        # Compute annotations based on ERP projection synthesized from multiple images.
        else:
            for face_i, face_ann_info in enumerate(new_meta['face_anns']):
                face_meta = new_meta['face_metas'][face_i]
                for face_ann in face_ann_info['annotations']:
                    if face_ann['iscrowd'] or face_ann['id'] in image_ann_dict:
                        continue
                    new_anns = _anns_from_mask(ann=face_ann, meta=face_meta, extend_type=extend_type)            
                    lines = []
                    for item in new_anns:
                        if not item['orientation'].startswith('F_'):
                            continue
                        
                        # Fix image id.
                        item['face_image_id'] = item['image_id']
                        item['image_id'] = new_meta['id']
                        
                        # Fix annotations.
                        sbox = item['bbox']
                        # B
                        if face_i == 0:
                            item['orientation'] = item['orientation'].replace('F_', 'B_')
                            if sbox[0] >= 0:
                                sbox = [sbox[0] - torch.pi, sbox[1], sbox[2], sbox[3], sbox[4]]
                            else:
                                sbox = [sbox[0] + torch.pi, sbox[1], sbox[2], sbox[3], sbox[4]]
                        # L
                        elif face_i == 1:
                            item['orientation'] = item['orientation'].replace('F_', 'L_')
                            sbox = [sbox[0] - torch.pi/2, sbox[1], sbox[2], sbox[3], sbox[4]]
                        # R
                        elif face_i == 2:
                            item['orientation'] = item['orientation'].replace('F_', 'R_')
                            sbox = [sbox[0] + torch.pi/2, sbox[1], sbox[2], sbox[3], sbox[4]]
                        # U
                        elif face_i == 3:
                            item['orientation'] = item['orientation'].replace('F_', 'U_')
                            sboxes = SphereBoxes([sbox])
                            sboxes.rotate(0., torch.pi / 2, 0.)
                            sbox = sboxes.tensor.tolist()[0]
                        # D
                        elif face_i == 4:
                            item['orientation'] = item['orientation'].replace('F_', 'D_')
                            sboxes = SphereBoxes([sbox])
                            sboxes.rotate(0., -torch.pi / 2, 0.)
                            sbox = sboxes.tensor.tolist()[0]
                        item['bbox'] = sbox
                        
                        lines.append(json.dumps(item) + '\n')
                    ann_f.writelines(lines)
                    ann_f.flush()
                
            for ann in ann_info['annotations']:
                progress_items[2] += 1
                if ann['iscrowd'] or ann['id'] in image_ann_dict:
                    continue
                
                new_anns = _anns_from_mask(ann=ann, meta=new_meta, extend_type=extend_type)            
                lines = []
                for item in new_anns:
                    if not item['orientation'].startswith('F_'):
                        continue
                    lines.append(json.dumps(item) + '\n')
                ann_f.writelines(lines)
                ann_f.flush()
            
        # print log
        t2 = time.time()
        delta_time['label'] += t2 - t1
        progress_items[0] += 1
        progress_items[5] += delta_time['read']
        progress_items[6] += delta_time['write']
        progress_items[7] += delta_time['pad']
        progress_items[8] += delta_time['transform']
        progress_items[9] += delta_time['label']
        if progress_items[0] % batch_size == 0:
            progress_items[4] = time.time() - time_start
            print(progress_bar % tuple(progress_items), flush=True)
          
    progress_items[4] = time.time() - time_start
    print(progress_bar % tuple(progress_items), flush=True)
    
    image_f.close()
    ann_f.close()


def main(
    dest_dir: str,
    src_dir: str,
    configs: List[dict],
    process_n: int,
    process_idxes: List[int],
    extend_type: ExtendType,
    random_faces: bool,
    classes: List[str] = None,
    width: int = 1920,
    height: int = 960,
):
    """ Transform COCO dataset to panoramic images. """
    # multiprocessing.set_start_method('spawn')
    pool = Pool(len(process_idxes))
    
    for config in configs:
        with open(os.path.join(src_dir, config['ann_file']), 'r') as f:
            anns = json.load(f)
        
        for idx, ann_infos in enumerate(_slice_anns(
            anns=anns,
            process_n=process_n,
            classes=classes,
            random_faces=random_faces
        )):
            if idx in process_idxes:
                pool.apply_async(_process_ann_info, kwds={
                    'process_idx': idx,
                    'ann_infos': ann_infos,
                    'config': config,
                    'src_dir': src_dir,
                    'dest_dir': dest_dir,
                    'height': height,
                    'width': width,
                    'extend_type': extend_type,
                    'random_faces': random_faces,
                }, error_callback=lambda e: print(e))
    pool.close()
    pool.join()
    

if __name__ == '__main__':
    process_n = int(sys.argv[1])
    process_idxes = [int(item) for item in sys.argv[2].split(',')]
    extend_type = ExtendType.of(sys.argv[3].strip())
    random_faces = int(sys.argv[4])
    
    classes = (
        'backpack', 'bed', 'book', 'bottle', 'bowl', 
        'chair', 'clock', 'cup', 'keyboard', 'microwave', 
        'mouse', 'oven', 'person', 'potted plant', 'refrigerator', 
        'sink', 'toilet', 'tv', 'vase', 'wine glass'
    )
     
    dest_dir = f'data/coco_panorama_{extend_type}{"_random" if random_faces else ""}'
    src_dir = 'data/coco'
    configs = [
        {
            'ann_file': 'annotations/instances_train2017.json',
            'data_prefix': 'train2017',
        },
        {
            'ann_file': 'annotations/instances_val2017.json',
            'data_prefix': 'val2017', 
        },   
    ]
     
    main(
        dest_dir=dest_dir,
        src_dir=src_dir,
        configs=configs,
        # classes=classes,
        classes=None,
        width=1920,
        height=960,
        process_n=process_n,
        process_idxes=process_idxes,
        extend_type=extend_type,
        random_faces=random_faces,
    )
