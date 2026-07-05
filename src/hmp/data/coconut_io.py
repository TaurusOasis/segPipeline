"""COCONut panoptic dataset IO for sampling, GT masks, and manifest building."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from ..common.hashing import sha256_file
from ..schemas import MediaItem, StratificationTags

PERSON_CATEGORY_ID = 1


def rgb_to_segment_id(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert COCONut/COCO panoptic RGB mask to integer segment ids."""
    if mask_rgb.ndim == 2:
        return mask_rgb.astype(np.int32)
    mask_rgb = mask_rgb.astype(np.int32)
    return mask_rgb[:, :, 0] + 256 * mask_rgb[:, :, 1] + 256 * 256 * mask_rgb[:, :, 2]


@dataclass(frozen=True)
class CoconutPersonInstance:
    image_id: int
    image_file: str
    instance_index: int
    segment_id: int
    category_id: int
    area: int
    bbox_xyxy: list[int]
    mask: np.ndarray


@dataclass(frozen=True)
class CoconutSample:
    image_id: int
    image_path: Path
    mask_path: Path
    width: int
    height: int
    persons: tuple[CoconutPersonInstance, ...]


def load_panoptic_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected panoptic dict, got {type(data).__name__}")
    return data


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _extract_person(mask_rgb: np.ndarray, segment_id: int, meta: dict[str, Any], instance_index: int) -> CoconutPersonInstance:
    seg_map = rgb_to_segment_id(mask_rgb)
    binary = seg_map == segment_id
    return CoconutPersonInstance(
        image_id=int(meta["image_id"]),
        image_file=str(meta["file_name"]).replace(".png", ".jpg"),
        instance_index=instance_index,
        segment_id=segment_id,
        category_id=int(meta["category_id"]),
        area=int(meta.get("area", int(binary.sum()))),
        bbox_xyxy=_bbox_from_mask(binary),
        mask=binary,
    )


def iter_coconut_person_samples(
    *,
    json_path: Path,
    mask_dir: Path,
    image_root: Path,
    image_subdir: str = "val2017",
    limit: Optional[int] = None,
    seed: int = 42,
) -> Iterator[CoconutSample]:
    """Yield COCONut images that contain at least one person instance."""
    data = load_panoptic_json(json_path)
    images_by_id = {int(img["id"]): img for img in data["images"]}
    anns = list(data["annotations"])
    if limit is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(anns), size=min(limit, len(anns)), replace=False)
        anns = [anns[i] for i in sorted(idx.tolist())]

    seen = 0
    for ann in anns:
        image_id = int(ann["image_id"])
        img_meta = images_by_id[image_id]
        image_file = str(img_meta["file_name"])
        image_path = image_root / image_subdir / image_file
        mask_path = mask_dir / str(ann["file_name"])
        if not image_path.exists() or not mask_path.exists():
            continue
        from PIL import Image

        mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
        person_segments = [
            s for s in ann["segments_info"]
            if int(s["category_id"]) == PERSON_CATEGORY_ID and int(s.get("isthing", 1)) == 1
        ]
        if not person_segments:
            continue
        persons = tuple(
            _extract_person(mask_rgb, int(seg["id"]), {**seg, "image_id": image_id, "file_name": ann["file_name"]}, i)
            for i, seg in enumerate(person_segments)
        )
        yield CoconutSample(
            image_id=image_id,
            image_path=image_path,
            mask_path=mask_path,
            width=int(img_meta["width"]),
            height=int(img_meta["height"]),
            persons=persons,
        )
        seen += 1
        if limit is not None and seen >= limit:
            break


def sample_to_media_item(sample: CoconutSample, *, source_dataset: str = "coconut") -> MediaItem:
    item_id = sample.image_path.stem
    tags = [source_dataset, "person"]
    if len(sample.persons) > 1:
        tags.append("multi_person")
    strat = StratificationTags(multi_person=len(sample.persons) > 1, occlusion="partial" if len(sample.persons) > 1 else "none")
    return MediaItem(
        item_id=item_id,
        media_type="image",
        path=str(sample.image_path),
        width=sample.width,
        height=sample.height,
        sha256=sha256_file(sample.image_path),
        tags=tags,
        source_dataset=source_dataset,
        stratification=strat,
        license_meta={"dataset": source_dataset, "split": "relabeled_coco_val", "license": "verify_before_use"},
    )
