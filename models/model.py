import timm
from transformers import AutoModel
import torch
import torch.nn as nn


class ViTCLSAdapter(nn.Module):
    """Adapt token-output ViT backbones to a plain embedding tensor (B, D) using CLS token."""

    def __init__(self, backbone: nn.Module, embedding_size: int):
        super().__init__()
        self.backbone = backbone
        self.embedding_size = embedding_size

    def _extract_last_hidden_state(self, outputs):
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, dict) and "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        if torch.is_tensor(outputs):
            return outputs
        raise ValueError(
            f"Unsupported ViT output type: {type(outputs)}. "
            "Expected tensor or object/dict with `last_hidden_state`."
        )

    def forward(self, x):
        outputs = self.backbone(x)
        hidden = self._extract_last_hidden_state(outputs)

        if not torch.is_tensor(hidden):
            raise ValueError(f"ViT hidden state must be a tensor, got: {type(hidden)}")

        if hidden.ndim == 3:
            # CLS token from token sequence: (B, N, D) -> (B, D)
            embeddings = hidden[:, 0, :]
        elif hidden.ndim == 2:
            # Already pooled.
            embeddings = hidden
        else:
            raise ValueError(f"Unsupported ViT hidden state shape: {tuple(hidden.shape)}")

        if embeddings.shape[-1] != self.embedding_size:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_size}, got {embeddings.shape[-1]}"
            )
        return embeddings


def get_model(MODEL_TYPE):
    # Download MegaDescriptor-T backbone from HuggingFace Hub
    if MODEL_TYPE == 'megadescriptor':
        backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
        arch = 'swin'
        patch_size = None
        number_of_patches = None
        embedding_size = 768
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 224
    elif MODEL_TYPE == 'lynx_megadescriptorV3':
        backbone = timm.create_model('hf-hub:strakajk/LynxV3-MegaDescriptor-T-224', num_classes=0, pretrained=True)
        arch = 'swin'
        patch_size = None
        number_of_patches = None
        embedding_size = 768
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 224
    elif MODEL_TYPE == 'lynx_megadescriptorV4':
        backbone = timm.create_model('hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256', num_classes=0, pretrained=True)
        arch = 'swin'
        patch_size = None
        number_of_patches = None
        embedding_size = 768
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 256
    elif MODEL_TYPE == 'miewid':
        backbone = AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)
        arch = 'cnn'
        patch_size = None
        number_of_patches = None
        embedding_size = 2152
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img_size = 440
    elif MODEL_TYPE == 'dinov2':
        pretrained_model_name = "facebook/dinov2-with-registers-small"
        raw_backbone = AutoModel.from_pretrained(pretrained_model_name)
        embedding_size = 384
        backbone = ViTCLSAdapter(raw_backbone, embedding_size=embedding_size)
        arch = 'vit'
        patch_size = 14
        img_size = 224
        number_of_patches = int((img_size // patch_size) ** 2)
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    elif MODEL_TYPE == 'dinov3':
        pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        raw_backbone = AutoModel.from_pretrained(pretrained_model_name)
        embedding_size = 384
        backbone = ViTCLSAdapter(raw_backbone, embedding_size=embedding_size)
        arch = 'vit'
        patch_size = 16
        img_size = 224
        number_of_patches = int((img_size // patch_size) ** 2)
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported")
    return backbone, embedding_size, mean, std, img_size, arch, patch_size, number_of_patches
