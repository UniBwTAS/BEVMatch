# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

@DATASETS.register_module()
class NuScenesDatasetKeypoints(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    #use init function from super class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_ranges = [
                 {'name': '5m10d', 'dist_range': (0, 5), 'max_angle': 10},
                 {'name': '10m30d', 'dist_range': (5, 10), 'max_angle': 30},
                 {'name': '20m50d', 'dist_range': (10, 20), 'max_angle': 50}
             ]


    def _filter_with_mask(self, ann_info: dict) -> dict:
       return super()._filter_with_mask(ann_info)

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)

        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        return super().parse_data_info(info)


    def _find_random_sample_in_range(self, idx: int, init_stamp, init_pose,
                                     min_dist: float, max_dist: float,
                                     max_angle: float) -> int:
        """Findet ein RANDOM sample innerhalb eines Distanz- und Winkelbereichs

        Args:
            idx: Start-Index
            init_stamp: Timestamp des Start-Frames
            init_pose: Pose des Start-Frames (4x4 transformation matrix)
            min_dist: Minimale Distanz in Metern
            max_dist: Maximale Distanz in Metern
            max_angle: Maximaler Rotationswinkel in Grad

        Returns:
            Index eines random Samples im Bereich (oder idx falls keines gefunden)
        """
        valid_candidates = []

        # Suche vorwärts
        for offset in range(1, min(150, len(self) - idx)):
            candidate_idx = idx + offset
            data_info = self.get_data_info(candidate_idx)

            candidate_stamp = data_info['timestamp']
            candidate_pose = np.array(data_info['ego2global'])

            # Zeit-Check
            if abs(init_stamp - candidate_stamp) > 20:
                break

            # Distanz berechnen
            angle, t = self.get_transformation_matrix(init_pose, candidate_pose)
            dist = np.linalg.norm(t)

            # Check ob im gewünschten Distanz-Bereich
            if min_dist <= dist <= max_dist:
                # Check ob Winkel OK
                if abs(angle) <= max_angle:
                    valid_candidates.append(candidate_idx)

        # Suche auch rückwärts
        # nein! keine suche rückwärts, da sonst die gleichen samples wie vorwärts gefunden werden
        """
        for offset in range(1, min(150, idx + 1)):
            candidate_idx = idx - offset
            if candidate_idx < 0:
                break

            data_info = self.get_data_info(candidate_idx)

            candidate_stamp = data_info['timestamp']
            candidate_pose = np.array(data_info['ego2global'])

            # Zeit-Check
            if abs(init_stamp - candidate_stamp) > 20:
                break

            # Distanz berechnen
            translation_diff = init_pose[:3, 3] - candidate_pose[:3, 3]
            dist = np.linalg.norm(translation_diff)

            # Check ob im gewünschten Distanz-Bereich
            if min_dist <= dist <= max_dist:
                # Winkel berechnen
                angle = self._compute_rotation_angle(init_pose, candidate_pose)

                # Check ob Winkel OK
                if angle <= max_angle:
                    valid_candidates.append(candidate_idx)
        """

        # Random sample aus valid candidates ziehen
        if len(valid_candidates) > 0:
            return np.random.choice(valid_candidates)
        else:
            # Fallback: current frame wenn nichts gefunden
            return idx


    def get_transformation_matrix(self, ego2global_A, ego2global_B):
        # Compute relative transformation (B in A's coordinate frame)
        relative_transformation = np.linalg.inv(ego2global_A) @ ego2global_B

        # Project to 2D (zero out z-components)
        orthogonal_projection = np.eye(4)
        orthogonal_projection[2, 2] = 0
        relative_transformation = orthogonal_projection @ relative_transformation

        # Extract rotation (3x3) and translation (3x1)
        R = relative_transformation[:3, :3]
        t = relative_transformation[:3, 3]

        # Scale translation by BEV resolution
        # Translation in grid coordinates (2D)

        # Rotation in grid coordinates (2x2)
        angle_rad = np.arctan2(R[1, 0], R[0, 0])
        angle_deg = np.degrees(angle_rad)

        return angle_deg, t


    def _compute_rotation_angle(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """Berechnet den Rotationswinkel zwischen zwei Posen in Grad

        Args:
            pose1: 4x4 Transformationsmatrix
            pose2: 4x4 Transformationsmatrix

        Returns:
            Rotationswinkel in Grad
        """
        # Rotationsmatrizen extrahieren
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]

        # Relative Rotation berechnen
        R_rel = R1.T @ R2

        # Rotationswinkel aus der Rotationsmatrix extrahieren
        # trace(R) = 1 + 2*cos(theta)
        trace = np.trace(R_rel)

        # Numerische Stabilität sicherstellen
        trace = np.clip(trace, -1.0, 3.0)

        # Winkel in Radiant berechnen
        theta_rad = np.arccos((trace - 1.0) / 2.0)

        # In Grad umwandeln
        theta_deg = np.degrees(theta_rad)

        return theta_deg


    def getNextValid(self, idx: int, init_stamp, init_pose) -> int:
        next_idx = idx + 1
        nextValid = next_idx < len(self)
        if nextValid:

            data_info = self.get_data_info(next_idx)
            next_stamp = data_info['timestamp']
            next_pose = np.array(data_info['ego2global'])

            #check if under 20sec and within 10m
            time_valid =  abs(init_stamp - next_stamp) < 20
            dist_valid = np.linalg.norm(init_pose[:3, 3] - next_pose[:3, 3]) < 15

            if time_valid and dist_valid:
                return next_idx

        return idx

    def getMaxNextValid(self, idx: int, n: int) -> int:
        next_idx = idx

        data_info = self.get_data_info(idx)
        init_pose = np.array(data_info['ego2global'])
        init_stamp = data_info['timestamp']

        for _ in range(n):
            next_idx = self.getNextValid(next_idx, init_stamp, init_pose)
            if next_idx == idx:
                break
            idx = next_idx
        return next_idx

    def getPrevValid(self, idx: int, init_stamp, init_pose) -> int:
        prev_idx = idx - 1
        prevValid = prev_idx >= 0
        if prevValid:
            data_info = self.get_data_info(prev_idx)
            prev_stamp = data_info['timestamp']
            prev_pose = np.array(data_info['ego2global'])

            #check if under 20sec and within 10m
            time_valid =  abs(init_stamp - prev_stamp) < 20
            dist_valid = np.linalg.norm(init_pose[:3, 3] - prev_pose[:3, 3]) < 15

            if time_valid and dist_valid:
                return prev_idx

        return idx
    def getMaxPrevValid(self, idx: int, n: int) -> int:
        prev_idx = idx
        data_info = self.get_data_info(idx)
        init_pose = np.array(data_info['ego2global'])
        init_stamp = data_info['timestamp']

        for _ in range(n):
            prev_idx = self.getPrevValid(prev_idx, init_stamp, init_pose)
            if prev_idx == idx:
                break
            idx = prev_idx
        return prev_idx

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th, idx-1th and idx+1th image and data information of dataset after
            ``self.pipeline``.
        """
        #get range of possible next indices
        next_idx = self.getMaxNextValid(idx, 40)
        prev_idx = self.getMaxPrevValid(idx, 40)
        #get random index between prev and next

        other_idx = np.random.randint(prev_idx, next_idx + 1)

        #create empty dict with None values
        data = {}

        #add data to dict
        data['current'] = super().__getitem__(idx)
        data['other'] = super().__getitem__(other_idx)
        data['other']['inputs']['valid'] = True
        data['current']['inputs']['valid'] = True



        # in test mode, we want to keep other frames also for evaluation
        # c) Zusätzliche samples nur bei Evaluation
        if self.test_mode:
            data_info = self.get_data_info(idx)
            init_pose = np.array(data_info['ego2global'])
            init_stamp = data_info['timestamp']

            for sample_config in self.eval_sample_ranges:
                sample_idx = self._find_random_sample_in_range(
                    idx, init_stamp, init_pose,
                    min_dist=sample_config['dist_range'][0],
                    max_dist=sample_config['dist_range'][1],
                    max_angle=sample_config['max_angle']
                )
                if sample_idx != idx:
                    data[f"other_{sample_config['name']}"] = super().__getitem__(sample_idx)
                    data[f"other_{sample_config['name']}"]['inputs']['valid'] = True
                else:
                    #its the current
                    data[f"other_{sample_config['name']}"] = data['current']
                    data[f"other_{sample_config['name']}"]['inputs']['valid'] = False

        return data

