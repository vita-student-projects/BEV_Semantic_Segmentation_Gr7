# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import pyquaternion
import imageio

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

import pdb


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', cover=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        if cover is not None:
            assert isinstance(cover, dict)
            self.cover = cover.pop('cover')
            self.cover_value = cover.pop('cover_value')
            self.cover_w = cover.pop('cover_w')


    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']

        # img is of shape (h, w, c, num_views)
        if not self.cover:
            img = np.stack(
                [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        else:
            assert '/CAM_FRONT/' in filename[0]
            img = []
            img.append(mmcv.imread(filename[0], self.color_type)) # (900, 1600, 3)
            for _ in range(5):
                img.append(np.full((900, 1600, 3),self.cover_value, dtype=img[0].dtype))

            if self.cover_w > 0:
                # front_right
                img[1][:, 0:self.cover_w, :] = img[0][:, -self.cover_w:, :]
                # front_left
                img[2][:, -self.cover_w:, :] = img[0][:, 0:self.cover_w, :]

            img = np.stack(img, axis=-1)

        # ['./data/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281440262460.jpg', 
        # './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281440270339.jpg', 
        # './data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281440254844.jpg', 
        # './data/nuscenes/samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281440287525.jpg', 
        # './data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281440297423.jpg', 
        # './data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281440277893.jpg']
        
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_MTL(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, is_train=False, using_ego=False, temporal_consist=False,
                 data_aug_conf={
                     'resize_lim': (0.193, 0.225),
                     'final_dim': (128, 352),
                     'rot_lim': (-5.4, 5.4),
                     'H': 900, 'W': 1600,
                     'rand_flip': True,
                     'bot_pct_lim': (0.0, 0.22),
                     'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 5,
                 }, load_seg_gt=False, num_seg_classes=14, select_classes=None):

        self.is_train = is_train
        self.using_ego = using_ego
        self.data_aug_conf = data_aug_conf
        self.load_seg_gt = load_seg_gt
        self.num_seg_classes = num_seg_classes
        self.select_classes = range(
            num_seg_classes) if select_classes is None else select_classes

        self.temporal_consist = temporal_consist
        self.test_time_augmentation = self.data_aug_conf.get('test_aug', False)

    def sample_augmentation(self, specify_resize=None, specify_flip=None):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            crop_h = max(0, newH - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize = resize + 0.04
            if specify_resize is not None:
                resize = specify_resize

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = max(0, newH - fH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if specify_flip is None else specify_flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
            cyclist = self.data_aug_conf.get('cyclist', False)
            if cyclist:
                start_id = np.random.choice(np.arange(len(cams)))
                cams = cams[start_id:] + cams[:start_id]
        return cams

    def get_img_inputs(self, results, specify_resize=None, specify_flip=None):
        img_infos = results['img_info']

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        cams = self.choose_cams()
        if self.temporal_consist:
            cam_augments = {}
            for cam in cams:
                cam_augments[cam] = self.sample_augmentation(
                    specify_resize=specify_resize, specify_flip=specify_flip)

        for frame_id, img_info in enumerate(img_infos):
            imgs.append([])
            rots.append([])
            trans.append([])
            intrins.append([])
            post_rots.append([])
            post_trans.append([])

            for cam in cams:
                cam_data = img_info[cam]
                filename = cam_data['data_path']
                filename = os.path.join(
                    results['data_root'], filename.split('nuscenes/')[1])

                img = Image.open(filename)

                # img = imageio.imread(filename)
                # img = Image.fromarray(img)

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                intrin = torch.Tensor(cam_data['cam_intrinsic'])
                # extrinsics
                rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
                tran = torch.Tensor(cam_data['sensor2lidar_translation'])

                # 进一步转换到 LiDAR 坐标系
                if self.using_ego:
                    cam2lidar = torch.eye(4)
                    cam2lidar[:3, :3] = torch.Tensor(
                        cam_data['sensor2lidar_rotation'])
                    cam2lidar[:3, 3] = torch.Tensor(
                        cam_data['sensor2lidar_translation'])

                    lidar2ego = torch.eye(4)
                    lidar2ego[:3, :3] = results['lidar2ego_rots']
                    lidar2ego[:3, 3] = results['lidar2ego_trans']

                    cam2ego = lidar2ego @ cam2lidar

                    rot = cam2ego[:3, :3]
                    tran = cam2ego[:3, 3]

                # augmentation (resize, crop, horizontal flip, rotate)
                if self.temporal_consist:
                    resize, resize_dims, crop, flip, rotate = cam_augments[cam]
                else:
                    # generate augmentation for each time-step, each
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                        specify_resize=specify_resize, specify_flip=specify_flip)

                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs[frame_id].append(normalize_img(img))
                intrins[frame_id].append(intrin)
                rots[frame_id].append(rot)
                trans[frame_id].append(tran)
                post_rots[frame_id].append(post_rot)
                post_trans[frame_id].append(post_tran)

        # [num_seq, num_cam, ...]
        imgs = torch.stack([torch.stack(x, dim=0) for x in imgs], dim=0)
        rots = torch.stack([torch.stack(x, dim=0) for x in rots], dim=0)
        trans = torch.stack([torch.stack(x, dim=0) for x in trans], dim=0)
        intrins = torch.stack([torch.stack(x, dim=0) for x in intrins], dim=0)
        post_rots = torch.stack([torch.stack(x, dim=0)
                                for x in post_rots], dim=0)
        post_trans = torch.stack([torch.stack(x, dim=0)
                                 for x in post_trans], dim=0)

        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        if (not self.is_train) and self.test_time_augmentation:
            results['flip_aug'] = []
            results['scale_aug'] = []
            img_inputs = []
            for flip in self.data_aug_conf.get('tta_flip', [False, ]):
                for scale in self.data_aug_conf.get('tta_scale', [None, ]):
                    results['flip_aug'].append(flip)
                    results['scale_aug'].append(scale)
                    img_inputs.append(
                        self.get_img_inputs(results, scale, flip))

            results['img_inputs'] = img_inputs
        else:
            results['img_inputs'] = self.get_img_inputs(results)

        return results


@PIPELINES.register_module()
class LoadAnnotations3D_MTL(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_instance_tokens=False,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)

        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype
        self.with_instance_tokens = with_instance_tokens


    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results


    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results


    # added
    def _load_instance_tokens(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        instance_tokens = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                instance_tokens.append(ann_info['instance_tokens'])
            else:
                instance_tokens.append(None)
        results['instance_tokens'] = instance_tokens

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)

        # 即使 without 3d bounding boxes, 也需要正常训练
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)

        if self.with_label_3d:
            results = self._load_labels_3d(results)

        if self.with_instance_tokens:
            results = self._load_instance_tokens(results)

        # loading valid_flags
        gt_valid_flags = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                gt_valid_flags.append(ann_info['gt_valid_flag'])
            else:
                gt_valid_flags.append(None)
        results['gt_valid_flag'] = gt_valid_flags

        # loading visibility tokens
        gt_vis_tokens = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                gt_vis_tokens.append(None)
        results['gt_vis_tokens'] = gt_vis_tokens

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
