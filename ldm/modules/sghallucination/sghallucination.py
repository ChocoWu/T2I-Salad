import logging
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from einops import repeat
from omegaconf import DictConfig
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModel
from .helpers.util import sparse_to_dense, tokenize_text_prompt
from .helpers.layout_tokenizer import LayoutSequenceTokenizer
from .base_model import BaseModel
from .categorical_diffusion.constrained import (
    ConstrainedMaskAndReplaceDiffusion,
)
from .categorical_diffusion.vanilla import VanillaMaskAndReplaceDiffusion
from .common.nn_lib import CustomDataParallel
from .common.util import shrink

logger = logging.getLogger(__name__)

Q_TYPES = {
    "vanilla": VanillaMaskAndReplaceDiffusion,
    "constrained": ConstrainedMaskAndReplaceDiffusion,
}


class SGHallucination(BaseModel):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        text_tokenizer: Optional[AutoTokenizer] = None,
        text_prompt_encoder: Optional[CLIPTextModel] = None,
        text_condition_seq_len: int = 30,
        transformer_type: str = "flattened",
        pos_emb: str = "elem_attr",
        num_timesteps: int = 100,
        auxiliary_loss_weight: float = 1e-1,
        q_type: str = "single",
        seq_type: str = "poset",
        **kwargs,
    ) -> None:
        super().__init__()
        assert q_type in Q_TYPES
        assert seq_type in ["set", "poset"]

        self.backbone_cfg = backbone_cfg
        self.pos_emb = pos_emb
        self.seq_type = seq_type
        # make sure MASK is the last vocabulary
        assert tokenizer.id_to_name(tokenizer.N_total - 1) == "mask"

        # tokenize text prompt via CLIPTokenizer
        self.text_tokenizer = text_tokenizer
        self.text_condition_seq_len = text_condition_seq_len
        # self.text_prompt_encoder = text_prompt_encoder

        # Note: make sure learnable parameters are inside self.model
        self.tokenizer = tokenizer
        model = Q_TYPES[q_type]

        self.model = model(
                backbone_cfg=shrink(backbone_cfg, 29 / 32),  # for fair comparison
                num_classes=tokenizer.N_total,
                max_token_length=tokenizer.max_token_length,
                num_timesteps=num_timesteps,
                pos_emb=pos_emb,
                transformer_type=transformer_type,
                auxiliary_loss_weight=auxiliary_loss_weight,
                tokenizer=tokenizer,
                text_prompt_encoder=text_prompt_encoder,
                **kwargs,
            )

        self.apply(self._init_weights)
        self.compute_stats()

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        if self.backbone_cfg.encoder_layer.text_condition:
            outputs, losses = self.model(inputs["seq"], text_prompt=inputs["text_ids"],
                                         text_prompt_mask=inputs["text_attention_mask"])
        else:
            outputs, losses = self.model(inputs["seq"])

        # aggregate losses for multi-GPU mode (no change in single GPU mode)
        new_losses = {k: v.mean() for (k, v) in losses.items()}

        return outputs, new_losses

    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        text_prompt: Optional[Tensor] = None,
        text_prompt_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        ids = self.model.sample(
            batch_size=batch_size, cond=cond, sampling_cfg=sampling_cfg,
            text_prompt=text_prompt, text_prompt_mask=text_prompt_mask,
            **kwargs
        ).cpu()
        layouts = self.tokenizer.decode(ids)
        if text_prompt is not None:
            captions = self.text_tokenizer.batch_decode(text_prompt)
            layouts['captions'] = captions
        return layouts

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: DictConfig
    ) -> DictConfig:
        sampling_cfg = super().aggregate_sampling_settings(sampling_cfg, args)
        if args.time_difference > 0:
            sampling_cfg.time_difference = args.time_difference

        return sampling_cfg

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        # print(bbox[:3])
        # print(label[:3])
        # print(mask[:3])
        inputs = {"label": label, "mask": mask, "bbox": bbox}
        ids = self.tokenizer.encode(inputs)

        if self.seq_type == "set":
            # randomly shuffle [PAD]'s location
            B, S = ids["mask"].size()
            C = self.tokenizer.N_var_per_element
            for i in range(B):
                indices = torch.randperm(S // C)
                indices = repeat(indices * C, "b -> (b c)", c=C)
                indices += torch.arange(S) % C
                for k in ids:
                    ids[k][i, :] = ids[k][i, indices]
        if 'captions' in batch.attr.keys():
            text_ids, text_attention_mask = tokenize_text_prompt(batch, self.text_tokenizer, self.text_condition_seq_len)
            ids['text_ids'] = text_ids
            ids["text_attention_mask"] = text_attention_mask
        return ids

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        # original code
        # base = "model.module.transformer.pos_emb"
        # additional_no_decay = [
        #     f"{base}.{name}"
        #     for name in self.model.module.transformer.pos_emb.no_decay_param_names
        # ]
        base = "model.transformer.pos_emb"
        additional_no_decay = [
            f"{base}.{name}"
            for name in self.model.transformer.pos_emb.no_decay_param_names
        ]
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
