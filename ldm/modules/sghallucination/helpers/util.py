import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch import BoolTensor, FloatTensor, LongTensor


import random
from enum import IntEnum
from itertools import combinations, product
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch import BoolTensor, FloatTensor, LongTensor
from torch_geometric.utils import to_dense_batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def batch_topk_mask(
    scores: FloatTensor,
    topk: LongTensor,
    mask: Optional[BoolTensor] = None,
) -> Tuple[BoolTensor, FloatTensor]:
    assert scores.ndim == 2 and topk.ndim == 1 and scores.size(0) == topk.size(0)
    if mask is not None:
        assert mask.size() == scores.size()
        assert (scores.size(1) >= topk).all()

    # ignore scores where mask = False by setting extreme values
    if mask is not None:
        const = -1.0 * float("Inf")
        const = torch.full_like(scores, fill_value=const)
        scores = torch.where(mask, scores, const)

    sorted_values, _ = torch.sort(scores, dim=-1, descending=True)
    topk = rearrange(topk, "b -> b 1")

    k_th_scores = torch.gather(sorted_values, dim=1, index=topk)

    topk_mask = scores > k_th_scores
    return topk_mask, k_th_scores


def batch_shuffle_index(
    batch_size: int,
    feature_length: int,
    mask: Optional[BoolTensor] = None,
) -> LongTensor:
    """
    Note: masked part may be shuffled because of unpredictable behaviour of sorting [inf, ..., inf]
    """
    if mask:
        assert mask.size() == [batch_size, feature_length]
    scores = torch.rand((batch_size, feature_length))
    if mask:
        scores[~mask] = float("Inf")
    _, indices = torch.sort(scores, dim=1)
    return indices



class RelSize(IntEnum):
    UNKNOWN = 0
    SMALLER = 1
    EQUAL = 2
    LARGER = 3


class RelLoc(IntEnum):
    UNKNOWN = 4
    LEFT = 5
    TOP = 6
    RIGHT = 7
    BOTTOM = 8
    CENTER = 9


REL_SIZE_ALPHA = 0.1


def detect_size_relation(b1, b2):
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    alpha = REL_SIZE_ALPHA
    if (1 - alpha) * a1 < a2 < (1 + alpha) * a1:
        return RelSize.EQUAL
    elif a1 < a2:
        return RelSize.LARGER
    else:
        return RelSize.SMALLER


def detect_loc_relation(b1, b2, is_canvas=False):
    if is_canvas:
        yc = b2[1]
        if yc < 1.0 / 3:
            return RelLoc.TOP
        elif yc < 2.0 / 3:
            return RelLoc.CENTER
        else:
            return RelLoc.BOTTOM

    else:
        l1, t1, r1, b1 = convert_xywh_to_ltrb(b1)
        l2, t2, r2, b2 = convert_xywh_to_ltrb(b2)

        if b2 <= t1:
            return RelLoc.TOP
        elif b1 <= t2:
            return RelLoc.BOTTOM
        elif r2 <= l1:
            return RelLoc.LEFT
        elif r1 <= l2:
            return RelLoc.RIGHT
        else:
            # might not be necessary
            return RelLoc.CENTER


def get_rel_text(rel, canvas=False):
    if type(rel) == RelSize:
        index = rel - RelSize.UNKNOWN - 1
        if canvas:
            return [
                "within canvas",
                "spread over canvas",
                "out of canvas",
            ][index]

        else:
            return [
                "larger than",
                "equal to",
                "smaller than",
            ][index]

    else:
        index = rel - RelLoc.UNKNOWN - 1
        if canvas:
            return [
                "",
                "at top",
                "",
                "at bottom",
                "at middle",
            ][index]

        else:
            return [
                "right to",
                "below",
                "left to",
                "above",
                "around",
            ][index]


# transform
class AddCanvasElement:
    x = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float)
    y = torch.tensor([0], dtype=torch.long)

    def __call__(self, data):
        flag = data.attr["has_canvas_element"].any().item()
        assert not flag
        if not flag:
            # device = data.x.device
            # x, y = self.x.to(device), self.y.to(device)
            data.x = torch.cat([self.x, data.x], dim=0)
            data.y = torch.cat([self.y, data.y + 1], dim=0)
            data.attr = data.attr.copy()
            data.attr["has_canvas_element"] = True
        return data


class AddRelationConstraints:
    def __init__(self, seed=None, edge_ratio=0.1, use_v1=False):
        self.edge_ratio = edge_ratio
        self.use_v1 = use_v1
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)

    def __call__(self, data):
        N = data.x.size(0)
        has_canvas = data.attr["has_canvas_element"]

        rel_all = list(product(range(2), combinations(range(N), 2)))
        size = int(len(rel_all) * self.edge_ratio)
        rel_sample = set(self.generator.sample(rel_all, size))

        edge_index, edge_attr = [], []
        rel_unk = 1 << RelSize.UNKNOWN | 1 << RelLoc.UNKNOWN
        for i, j in combinations(range(N), 2):
            bi, bj = data.x[i], data.x[j]
            canvas = data.y[i] == 0 and has_canvas

            if self.use_v1:
                if (0, (i, j)) in rel_sample:
                    rel_size = 1 << detect_size_relation(bi, bj)
                    rel_loc = 1 << detect_loc_relation(bi, bj, canvas)
                else:
                    rel_size = 1 << RelSize.UNKNOWN
                    rel_loc = 1 << RelLoc.UNKNOWN
            else:
                if (0, (i, j)) in rel_sample:
                    rel_size = 1 << detect_size_relation(bi, bj)
                else:
                    rel_size = 1 << RelSize.UNKNOWN

                if (1, (i, j)) in rel_sample:
                    rel_loc = 1 << detect_loc_relation(bi, bj, canvas)
                else:
                    rel_loc = 1 << RelLoc.UNKNOWN

            rel = rel_size | rel_loc
            if rel != rel_unk:
                edge_index.append((i, j))
                edge_attr.append(rel)

        data.edge_index = torch.as_tensor(edge_index).long()
        data.edge_index = data.edge_index.t().contiguous()
        data.edge_attr = torch.as_tensor(edge_attr).long()

        return data


class RandomOrder:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        device = data.x.device
        N = data.x.size(0)
        idx = torch.randperm(N, device=device)
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class SortByLabel:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        idx = data.y.sort().indices
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class LexicographicOrder:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        x, y, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(y, x)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class AddNoiseToBBox:
    def __init__(self, std: float = 0.05):
        self.std = float(std)

    def __call__(self, data):
        noise = torch.normal(0, self.std, size=data.x.size(), device=data.x.device)
        data.x_orig = data.x.clone()
        data.x = data.x + noise
        data.attr = data.attr.copy()
        data.attr["NoiseAdded"][0] = True
        return data


class HorizontalFlip:
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 0] = 1 - data.x[:, 0]
        return data



def compose_transform(transforms: List[str]) -> T.Compose:
    """
    Compose transforms, optionally with args (e.g., AddRelationConstraints(edge_ratio=0.1))
    """
    transform_list = []
    for t in transforms:
        if "(" in t and ")" in t:
            pass
        else:
            t += "()"
        transform_list.append(eval(t))
    return T.Compose(transform_list)


def sparse_to_dense(
    batch,
    device: torch.device = torch.device("cpu"),
    remove_canvas: bool = False,
    max_num_nodes: Optional[int] = None,
) -> Tuple[FloatTensor, LongTensor, BoolTensor, BoolTensor]:
    batch = batch.to(device)
    bbox, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_num_nodes)
    label, mask = to_dense_batch(batch.y, batch.batch, max_num_nodes=max_num_nodes)

    if remove_canvas:
        bbox = bbox[:, 1:].contiguous()
        label = label[:, 1:].contiguous() - 1  # cancel +1 effect in transform
        label = label.clamp(min=0)
        mask = mask[:, 1:].contiguous()

    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def tokenize_text_prompt_single(
    text: Union[str, List[str]],
    text_tokenizer,
    text_condition_seq_len,
    device: torch.device = torch.device("cpu"),
):
    """
    this fucntion is used to predict the single or list of text prompt.
    """
    if isinstance(text, str):
        prompt = [text]
    else:
        prompt = text
    text_inputs = text_tokenizer(
        prompt,
        padding="max_length",
        max_length=min(text_tokenizer.model_max_length, text_condition_seq_len),
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = text_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = text_tokenizer.batch_decode(
            untruncated_ids[:, text_tokenizer.model_max_length - 1: -1]
        )
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {text_tokenizer.model_max_length} tokens: {removed_text}"
        )
        # print(self.text_condition_seq_len)
        print(untruncated_ids.shape)
        print(text_input_ids.shape)
        # exit(0)
    return text_input_ids.to(device), text_inputs.attention_mask.to(device)


def tokenize_text_prompt(
        batch,
        text_tokenizer,
        text_condition_seq_len,
        device: torch.device = torch.device("cpu"),
):
    batch = batch.to(device)
    # tokenize the text prompt
    prompt = batch.attr['captions']
    # print("batch captions")
    # print(batch.attr['captions'])
    # print(self.text_condition_seq_len)
    text_inputs = text_tokenizer(
        prompt,
        padding="max_length",
        max_length=min(text_tokenizer.model_max_length, text_condition_seq_len),
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = text_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = text_tokenizer.batch_decode(
            untruncated_ids[:, text_tokenizer.model_max_length - 1: -1]
        )
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {text_tokenizer.model_max_length} tokens: {removed_text}"
        )
        # print(self.text_condition_seq_len)
        print(untruncated_ids.shape)
        print(text_input_ids.shape)
        # exit(0)
    return text_input_ids.to(device), text_inputs.attention_mask.to(device)


def loader_to_list(
    loader: torch.utils.data.dataloader.DataLoader,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    layouts = []
    for batch in loader:
        bbox, label, _, mask = sparse_to_dense(batch)
        for i in range(len(label)):
            valid = mask[i].numpy()
            layouts.append((bbox[i].numpy()[valid], label[i].numpy()[valid]))
    return layouts


def split_num_samples(N: int, batch_size: int) -> List[int]:
    quontinent = N // batch_size
    remainder = N % batch_size
    dataloader = quontinent * [batch_size]
    if remainder > 0:
        dataloader.append(remainder)
    return dataloader


if __name__ == "__main__":
    scores = torch.arange(6).view(2, 3).float()
    # topk = torch.arange(2) + 1
    topk = torch.full((2,), 3)
    mask = torch.full((2, 3), False)
    # mask[1, 2] = False
    print(batch_topk_mask(scores, topk, mask=mask))

