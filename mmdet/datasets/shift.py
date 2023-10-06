"""SHIFT dataset for mmdet.

This is a reference code for mmdet style dataset of the SHIFT dataset. Note that
only single-view 2D detection, instance segmentation, and tracking are supported.
Please refer to the torch version of the dataloader for multi-view multi-task cases.

The codes are tested in mmdet-2.20.0.

Notes
-----
1.  Please copy this file to `mmdet/datasets/` and update the `mmdet/datasets/__init__.py`
    so that the `SHIFTDataset` class is imported. You can refer to their official tutorial at
    https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html.
2.  The `backend_type` must be one of ['file', 'zip', 'hdf5'] and the `img_prefix`
    must be consistent with the backend_type.
3.  Since the images are loaded before the pipeline with the selected backend, there is no need
    to add a `LoadImageFromFile` module in the pipeline again.
4.  For instance segmentation please use the `det_insseg_2d.json` for the `ann_file`,
    and add a `LoadAnnotations(with_mask=True)` module in the pipeline.
"""

import json
import os
import csv

import mmcv
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import LoadAnnotations

from shift_dev.utils.backend import HDF5Backend, ZipBackend


@DATASETS.register_module()
class SHIFTDataset(CustomDataset):
    CLASSES = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")

    WIDTH = 1280
    HEIGHT = 800

    def __init__(self, *args, backend_type: str = "file", filter_cfg=None, seq_info='seq.csv', **kwargs):
        """Initialize the SHIFT dataset.

        Args:
            backend_type (str, optional): The type of the backend. Must be one of
                ['file', 'zip', 'hdf5']. Defaults to "file".
        """
        super().__init__(*args, **kwargs)
        self.backend_type = backend_type
        if backend_type == "file":
            self.backend = None
        elif backend_type == "zip":
            self.backend = ZipBackend()
        elif backend_type == "hdf5":
            self.backend = HDF5Backend()
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}! "
                "Must be one of ['file', 'zip', 'hdf5']"
            )
        self.seq_info = seq_info
        if filter_cfg is not None and 'attributes' in filter_cfg:
            self.attr = filter_cfg['attributes']
            self.filter_attributes()

    def load_annotations(self, ann_file):
        print("Loading annotations...")
        with open(ann_file, "r") as f:
            data = json.load(f)

        data_infos = []
        for img_info in data["frames"]:
            img_filename = os.path.join(
                self.img_prefix, img_info["videoName"], img_info["name"]
            )

            bboxes = []
            labels = []
            track_ids = []
            masks = []
            for label in img_info["labels"]:
                bbox = label["box2d"]
                bboxes.append((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
                labels.append(self.CLASSES.index(label["category"]))
                track_ids.append(label["id"])
                if "rle" in label and label["rle"] is not None:
                    masks.append(label["rle"])

            data_infos.append(
                dict(
                    filename=img_filename,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64),
                        track_ids=np.array(track_ids).astype(np.int64),
                        masks=masks if len(masks) > 0 else None,
                    ),
                )
            )
        self.img_prefix = None
        return data_infos

    def get_img(self, idx):
        filename = self.data_infos[idx]["filename"]
        if self.backend_type == "zip":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        elif self.backend_type == "hdf5":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        else:
            return mmcv.imread(filename)

    def get_img_info(self, idx):
        return dict(
            filename=self.data_infos[idx]["filename"],
            width=self.WIDTH,
            height=self.HEIGHT,
        )

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]

    def prepare_train_img(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        # Filter out images without annotations during training
        if len(ann_info["bboxes"]) == 0:
            return None
        results = dict(img=img, img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        results = dict(img=img, img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def filter_attributes(self):
        valid_inds = []
        reader = csv.reader(open(os.path.join(self.data_root, self.seq_info), 'r'), delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                header = row
                attr_idx = [min([idx for idx, nm in enumerate(header) if k in nm]) for k in list(self.attr.keys())]
                attr_vals = list(self.attr.values())
            else:
                valid = 1
                for i, v in zip(attr_idx, attr_vals):
                    if row[i] != v:
                        valid = 0
                if valid == 1:
                    valid_inds.append(row[0])
        print('valid dirs {} among {} after filtering'.format(len(valid_inds), idx))

        filtered_data_infos = []
        for d in self.data_infos:
            ind = os.path.basename(os.path.dirname(d['filename']))
            if ind in valid_inds:
                # new_d = {k: d[k] for k in d}
                # new_d['filename'] = os.path.join(ind, os.path.basename(d['filename']))
                filtered_data_infos.append(d)
        print('valid files {} among {} after filtering'.format(len(filtered_data_infos), len(self.data_infos)))
        self.data_infos = filtered_data_infos
        self._set_group_flag()


if __name__ == "__main__":
    """Example for loading the SHIFT dataset for instance segmentation."""

    dataset = SHIFTDataset(
        data_root="./SHIFT_dataset/discrete/images",
        ann_file="train/front/det_insseg_2d.json",
        img_prefix="train/front/img.zip",
        backend_type="zip",
        pipeline=[LoadAnnotations(with_mask=True)],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("img:", data["img"].shape)
        print("ann_info.bboxes:", data["ann_info"]["bboxes"].shape)
        print("ann_info.labels:", data["ann_info"]["labels"].shape)
        print("ann_info.track_ids:", data["ann_info"]["track_ids"].shape)
        if "gt_masks" in data:
            print("gt_masks:", data["gt_masks"])
        break