"""Tests for trimap generation (Step 17)."""

from __future__ import annotations

import numpy as np

from hmp.data.mask_io import read_binary_mask, write_binary_mask
from hmp.matting.trimap import make_trimap, make_trimap_from_annotation


def _read_uint8(path):
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def test_trimap_values_and_regions():
    m = np.zeros((100, 100), bool)
    m[20:80, 20:80] = True
    tri = make_trimap(m, radius=6)
    vals = set(np.unique(tri).tolist())
    assert vals == {0, 128, 255}
    # deep interior is foreground
    assert tri[50, 50] == 255
    # exterior far away is background
    assert tri[5, 5] == 0
    # boundary band is unknown (just outside the original mask edge)
    assert tri[18, 50] == 128 or tri[82, 50] == 128


def test_trimap_small_radius_still_has_band():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    tri = make_trimap(m, radius=2)
    assert 128 in np.unique(tri)


def test_trimap_empty_mask_all_background():
    tri = make_trimap(np.zeros((30, 30), bool), radius=4)
    assert set(np.unique(tri).tolist()) <= {0}


def test_make_trimap_from_annotation(tmp_path):
    from hmp.common.jsonl import write_jsonl
    from hmp.config import Config
    from hmp.schemas import AnnotationRecord, InstanceAnnotation

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask = np.zeros((60, 60), bool)
    mask[15:45, 15:45] = True
    write_binary_mask(mask_dir / "a_person_0.png", mask)
    ann = tmp_path / "ann.jsonl"
    write_jsonl(
        [AnnotationRecord(item_id="a", instances=[InstanceAnnotation(instance_id="person_0", bbox_xyxy=[15, 15, 45, 45], mask_path=str(mask_dir / "a_person_0.png"))])],
        ann,
        overwrite=True,
    )
    out_dir = tmp_path / "trimaps"
    cfg = Config({"paths": {"refined_annotation_path": str(ann)}, "trimap": {"output_dir": str(out_dir), "radius": 5}})
    out = make_trimap_from_annotation(cfg, project_root=tmp_path)
    assert out == out_dir
    tri = _read_uint8(out_dir / "a_person_0_trimap.png")
    assert set(np.unique(tri).tolist()) == {0, 128, 255}


def test_make_trimap_dry_run(tmp_path):
    from hmp.common.jsonl import write_jsonl
    from hmp.config import Config
    from hmp.schemas import AnnotationRecord

    ann = tmp_path / "ann.jsonl"
    write_jsonl([AnnotationRecord(item_id="a")], ann, overwrite=True)
    out_dir = tmp_path / "trimaps"
    cfg = Config({"paths": {"refined_annotation_path": str(ann)}, "trimap": {"output_dir": str(out_dir)}})
    make_trimap_from_annotation(cfg, project_root=tmp_path, dry_run=True)
    assert not out_dir.exists()