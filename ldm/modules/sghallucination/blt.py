import copy
import logging
import random
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModel

from trainer.data.util import sparse_to_dense, tokenize_text_prompt
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import sample
from trainer.helpers.task import duplicate_cond
from trainer.helpers.util import batch_topk_mask
from trainer.models.base_model import BaseModel
from trainer.models.common.nn_lib import (
    CategoricalTransformer,
    CustomDataParallel,
    SeqLengthDistribution,
)
from trainer.models.common.util import get_dim_model

logger = logging.getLogger(__name__)

TARGET_ATTRS = [["c"], ["w", "h"], ["x", "y"]]  # (category, size, position)


def sample_mask(mask: Tensor, n_attr: int = 1):
    num_true = mask.sum().item() * n_attr
    n = random.randint(1, num_true)
    x = [True for _ in range(n)] + [False for _ in range(num_true - n)]
    random.shuffle(x)
    x += [False for _ in range(mask.size(0) * n_attr - len(x))]
    return rearrange(Tensor(x).to(mask), "(s c) -> s c", c=n_attr)


class BLT(BaseModel):
    """
    To reproduce
    BLT: Bidirectional Layout Transformer for Controllable Layout Generation (ECCV2022)
    https://arxiv.org/abs/2112.05112
    """

    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        use_padding_as_vocab: bool = False,
        text_tokenizer: Optional[AutoTokenizer] = None,
        text_prompt_encoder: Optional[CLIPTextModel] = None,
        text_condition_seq_len: int = 77,
    ) -> None:
        super().__init__()
        # check conditions
        if use_padding_as_vocab:
            assert tokenizer.pad_until_max
        assert tokenizer.var_order == "c-x-y-w-h"

        self.tokenizer = tokenizer
        self.use_padding_as_vocab = use_padding_as_vocab

        # Note: make sure learnable parameters are inside self.model
        backbone = instantiate(backbone_cfg)
        # self.model = CustomDataParallel(
        #     CategoricalTransformer(
        #         backbone=backbone,
        #         dim_model=get_dim_model(backbone_cfg),
        #         num_classes=tokenizer.N_total,
        #         max_token_length=tokenizer.max_token_length,
        #     )
        # )
        self.backbone_cfg = backbone_cfg
        self.text_tokenizer = text_tokenizer
        self.text_condition_seq_len = text_condition_seq_len
        self.model = CategoricalTransformer(
                    backbone=backbone,
                    dim_model=get_dim_model(backbone_cfg),
                    num_classes=tokenizer.N_total,
                    max_token_length=tokenizer.max_token_length,
                    text_prompt_encoder=text_prompt_encoder,
                )
        self.apply(self._init_weights)
        self.compute_stats()
        self.seq_dist = SeqLengthDistribution(tokenizer.max_seq_length)
        self.loss_fn_ce = nn.CrossEntropyLoss()

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        loss_mask = inputs["loss_mask"]

        if self.use_padding_as_vocab:
            if self.backbone_cfg.encoder_layer.text_condition:
                outputs = self.model(inputs["input"], text_prompt=inputs["text_ids"],
                                     text_prompt_mask=inputs["text_attention_mask"])
            else:
                outputs = self.model(inputs["input"])
            # outputs = self.model(inputs["input"])
        else:
            if self.backbone_cfg.encoder_layer.text_condition:
                outputs = self.model(inputs["input"], src_key_padding_mask=inputs["padding_mask"],
                                     text_prompt=inputs["text_ids"],
                                     text_prompt_mask=inputs["text_attention_mask"])
            else:
                outputs = self.model(inputs["input"], src_key_padding_mask=inputs["padding_mask"])
            # outputs = self.model(
            #     inputs["input"], src_key_padding_mask=inputs["padding_mask"]
            # )
        nll_loss = self.loss_fn_ce(
            outputs["logits"][loss_mask],
            inputs["target"][loss_mask],
        )
        losses = {"nll_loss": nll_loss}

        # replace masked tokens with predicted tokens
        outputs["outputs"] = copy.deepcopy(inputs["input"])
        ids = torch.argmax(outputs["logits"], dim=-1)
        outputs["outputs"][loss_mask] = ids[loss_mask]

        return outputs, losses

    def sample(
        self,
        batch_size: Optional[int],
        cond: Optional[Tensor] = None,
        sampling_cfg: Optional[DictConfig] = None,
        device: Optional[torch.device] = None,
        text_prompt: Optional[Tensor] = None,
        text_prompt_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Generate sample based on z.
        z can be either given or sampled from some distribution (e.g., normal)
        """

        mask_id = self.tokenizer.name_to_id("mask")
        pad_id = self.tokenizer.name_to_id("pad")

        B, S = (batch_size, self.tokenizer.max_token_length)

        if "num_timesteps" in sampling_cfg:
            self.num_timesteps = sampling_cfg.num_timesteps
        else:
            self.num_timesteps = 9
        assert self.num_timesteps % 3 == 0

        if cond:
            cond = duplicate_cond(cond, batch_size)
            seq = cond["seq"].clone()
            # **_user will not be updated (kept as reference)
            seq_user = cond["seq"].clone()
            mask_user = cond["mask"].clone()
            if not self.use_padding_as_vocab:
                src_key_padding_mask_user = seq == pad_id
        else:
            n_elements = self.seq_dist.sample(B) * self.tokenizer.N_var_per_element
            indices = rearrange(torch.arange(S), "s -> 1 s")
            mask = indices < rearrange(n_elements, "b -> b 1")
            seq = torch.full((B, S), fill_value=pad_id)
            seq[mask] = mask_id
            seq_user = seq.clone()
            mask_user = ~mask.clone()
            src_key_padding_mask_user = ~mask.clone()

        T = self.num_timesteps // 3
        n_attr = self.tokenizer.N_var_per_element
        indices = [
            [self.tokenizer.var_names.index(a) for a in attrs] for attrs in TARGET_ATTRS
        ]
        for target_attr_indices in indices:
            # ignore already filled region or non-target attributes
            attr_indices = repeat(torch.arange(S), "s -> b s", b=B) % n_attr
            keep_attr = torch.full((B, S), fill_value=True)
            for ind in target_attr_indices:
                keep_attr[attr_indices == ind] = False

            for t in range(T):
                ratio = (T - (t + 1)) / T
                if self.use_padding_as_vocab:
                    logits = self.model(seq.to(device),
                                        text_prompt=text_prompt.to(device),
                                        text_prompt_mask=text_prompt_mask.to(device))["logits"].cpu()
                else:
                    logits = self.model(
                        seq.to(device), src_key_padding_mask=src_key_padding_mask_user.to(device),
                        text_prompt=text_prompt.to(device), text_prompt_mask=text_prompt_mask.to(device)
                    )["logits"].cpu()

                invalid = repeat(~self.tokenizer.token_mask, "s c -> b s c", b=B)
                logits[invalid] = -float("Inf")

                seq_pred = sample(rearrange(logits, "b s c -> b c s"), sampling_cfg)
                confidence = torch.gather(
                    logits, -1, rearrange(seq_pred, "b 1 s -> b s 1")
                )
                confidence = rearrange(confidence, "b s 1 -> b s")
                seq_pred = rearrange(seq_pred, "b 1 s -> b s")

                # update by predicted tokens
                mask = (seq == mask_id) & (~keep_attr)
                seq = torch.where(mask, seq_pred, seq)

                if t < T - 1:
                    # re-fill [MASK] for unconfident predictions
                    n_elem = reduce(
                        ~(mask_user | keep_attr), "b s -> b", reduction="sum"
                    )
                    topk = (n_elem * ratio).long()
                    is_unconfident, _ = batch_topk_mask(
                        -1.0 * confidence, topk, mask=mask
                    )
                    seq[is_unconfident] = mask_id

                # make sure to use user-defined inputs
                seq = torch.where(mask_user, seq_user, seq)

        layouts = self.tokenizer.decode(seq)
        return layouts

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        self.seq_dist(mask)

        inputs = self.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        B = inputs["mask"].size(0)
        C = self.tokenizer.N_var_per_element
        S = inputs["mask"].size(1) // C
        mask_id = self.tokenizer._special_token_name_to_id["mask"]

        sampled_indices = torch.randint(0, len(TARGET_ATTRS), size=(B,))
        loss_mask = torch.full((B, S, C), False)
        for i, ind in enumerate(sampled_indices):
            if self.use_padding_as_vocab:
                # no constraint on mask location
                tmp_mask = torch.full((S,), True)
            else:
                tmp_mask = inputs["mask"][i, 0::C]
            if ind == 0:  # C(ategory)
                loss_mask[i, :, 0:1] = sample_mask(tmp_mask, n_attr=1)
            elif ind == 1:  # P(osition)
                loss_mask[i, :, 1:3] = sample_mask(tmp_mask, n_attr=2)
            elif ind == 2:  # S(ize)
                loss_mask[i, :, 3:] = sample_mask(tmp_mask, n_attr=2)
        loss_mask = rearrange(loss_mask, "b s c -> b (s c)")

        masked_seq = copy.deepcopy(inputs["seq"])
        masked_seq[loss_mask] = mask_id
        if 'captions' in batch.attr.keys():
            text_ids, text_attention_mask = tokenize_text_prompt(batch, self.text_tokenizer, self.text_condition_seq_len)
            # ids['text_ids'] = text_ids
            # ids["text_attention_mask"] = text_attention_mask
            return {
                "target": inputs["seq"],
                "padding_mask": ~inputs["mask"],
                "loss_mask": loss_mask,
                "input": masked_seq,
                "text_ids": text_ids,
                "text_attention_mask": text_attention_mask}
        return {
            "target": inputs["seq"],
            "padding_mask": ~inputs["mask"],
            "loss_mask": loss_mask,
            "input": masked_seq,
        }

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        return super().optim_groups(
            weight_decay=weight_decay,
            additional_no_decay=[
                "model.pos_emb.pos_emb",
            ],
        )
