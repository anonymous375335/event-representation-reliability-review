#!/usr/bin/env python3

import os
import sys
import types
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENET_REPO = Path(os.environ.get("EVENET_REPO", str(ROOT / "external" / "EveNet_Public")))


def _install_lightning_stub():
    if "lightning.pytorch.loggers" in sys.modules:
        return

    lightning = sys.modules.get("lightning", types.ModuleType("lightning"))
    lightning_pytorch = sys.modules.get("lightning.pytorch", types.ModuleType("lightning.pytorch"))
    lightning_loggers = sys.modules.get(
        "lightning.pytorch.loggers",
        types.ModuleType("lightning.pytorch.loggers"),
    )

    class WandbLogger:  # pragma: no cover - import stub only.
        pass

    lightning_loggers.WandbLogger = WandbLogger
    lightning_pytorch.loggers = lightning_loggers
    lightning.pytorch = lightning_pytorch

    sys.modules["lightning"] = lightning
    sys.modules["lightning.pytorch"] = lightning_pytorch
    sys.modules["lightning.pytorch.loggers"] = lightning_loggers


def build_evenet_passthrough_normalization(feature_dim, condition_dim, labels, device):
    num_classes = int(labels.max().item() + 1)
    return {
        "input_mean": {
            "Source": torch.zeros(feature_dim, dtype=torch.float32, device=device),
            "Conditions": torch.zeros(condition_dim, dtype=torch.float32, device=device),
        },
        "input_std": {
            "Source": torch.ones(feature_dim, dtype=torch.float32, device=device),
            "Conditions": torch.ones(condition_dim, dtype=torch.float32, device=device),
        },
        "input_num_mean": {"Source": torch.zeros(1, dtype=torch.float32, device=device)},
        "input_num_std": {"Source": torch.ones(1, dtype=torch.float32, device=device)},
        "class_counts": torch.bincount(labels, minlength=num_classes).float().to(device),
        "class_balance": torch.ones(num_classes, dtype=torch.float32, device=device),
    }


class EveNetOfficialTeacherAdapter(nn.Module):
    def __init__(self, checkpoint_path: Path, normalization_dict, device, repo_root: Path | None = None):
        super().__init__()
        self.repo_root = Path(repo_root) if repo_root is not None else DEFAULT_EVENET_REPO
        if not self.repo_root.exists():
            raise FileNotFoundError(f"EveNet repo root not found: {self.repo_root}")

        _install_lightning_stub()
        sys.path.insert(0, str(self.repo_root))

        from evenet.control.global_config import Config
        from evenet.network.evenet_model import EveNetModel
        from evenet.utilities.tool import safe_load_state

        cfg = Config()
        cfg.load_yaml(self.repo_root / "share" / "finetune-example.yaml")
        cfg.options.Training.Components.Classification.include = True
        cfg.options.Training.Components.Regression.include = False
        cfg.options.Training.Components.Assignment.include = False
        cfg.options.Training.Components.Segmentation.include = False
        cfg.options.Training.Components.GlobalGeneration.include = False
        cfg.options.Training.Components.ReconGeneration.include = False
        cfg.options.Training.Components.TruthGeneration.include = False

        model = EveNetModel(
            config=cfg,
            device=device,
            classification=True,
            regression=False,
            global_generation=False,
            point_cloud_generation=False,
            neutrino_generation=False,
            assignment=False,
            segmentation=False,
            normalization_dict=normalization_dict,
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        teacher_state = ckpt.get("ema_state_dict") or ckpt.get("state_dict")
        if teacher_state is None:
            raise ValueError("EveNet checkpoint does not contain `ema_state_dict` or `state_dict`.")
        safe_load_state(model, teacher_state, verbose=False)

        self.model = model
        self.hidden_dim = model.network_cfg.Classification.hidden_dim
        self.logit_key = next(iter(model.Classification.networks.keys()))

    def forward(self, features, valid_masks, conditions, conditions_mask):
        input_point_cloud = features
        input_point_cloud_mask = valid_masks.unsqueeze(-1)
        global_conditions = conditions.unsqueeze(1)
        global_conditions_mask = conditions_mask.view(-1, 1, 1).float()
        time = torch.zeros(features.shape[0], device=features.device)
        time_masking = torch.zeros_like(input_point_cloud_mask, dtype=torch.float32)

        input_point_cloud = self.model.sequential_normalizer(
            x=input_point_cloud,
            mask=input_point_cloud_mask,
        )
        global_conditions = self.model.global_normalizer(
            x=global_conditions,
            mask=global_conditions_mask,
        )

        embedded_global_conditions = self.model.GlobalEmbedding(
            x=global_conditions,
            mask=global_conditions_mask,
        )

        local_points = input_point_cloud[..., self.model.local_feature_indices]
        embedded_points = self.model.PET(
            input_features=input_point_cloud,
            input_points=local_points,
            mask=input_point_cloud_mask,
            attn_mask=None,
            time=time,
            time_masking=time_masking,
        )

        embeddings, _, event_token = self.model.ObjectEncoder(
            encoded_vectors=embedded_points,
            mask=input_point_cloud_mask,
            condition_vectors=embedded_global_conditions,
            condition_mask=global_conditions_mask,
        )

        class_token = self.model.Classification.class_transformer(
            x=embeddings,
            class_token=event_token,
            mask=input_point_cloud_mask,
        )
        class_token = event_token + class_token
        logits = self.model.Classification.networks[self.logit_key](class_token)
        return class_token, logits
