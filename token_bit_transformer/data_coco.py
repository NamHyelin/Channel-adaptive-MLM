# data_coco.py
from typing import List, Iterable
from datasets import load_dataset


def iter_coco_captions(
    dataset_name: str,
    split: str,
    text_field: str,
    max_samples: int,
) -> Iterable[str]:
    
    from data_COCO_download.coco_caption_dataset import CocoCaptionPairs

    # ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    ds = CocoCaptionPairs(
        root="coco_data",
        split=split,
        image_dir=None,      # 캡션만 쓸 거면 None
        flatten=False,       # 이미지당 랜덤 1캡션
    )

    
    captions: List[str] = []
    n = 0

    for ex in ds:
        if n >= max_samples:
            break

        cap = _extract_caption(ex, text_field)

        if isinstance(cap, str) and cap:
            captions.append(cap)
            n += 1

    # main이 list(iter_coco_captions(...)) 를 호출하므로
    # 여기서는 iterable로 내보내기만 하면 됨
    yield from captions



def _extract_caption(ex, text_field: str):
    # CocoCaptionPairs tuple
    if isinstance(ex, tuple) and len(ex) == 2:
        return ex[1]

    # HuggingFace dict
    if isinstance(ex, dict):
        cap = ex.get(text_field)
        if cap is None:
            for k in ["caption", "captions", "text"]:
                if k in ex:
                    cap = ex[k]
                    break
        if isinstance(cap, list) and cap:
            return str(cap[0])
        if isinstance(cap, str):
            return cap

    return None
