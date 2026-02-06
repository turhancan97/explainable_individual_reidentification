import timm
from transformers import AutoModel
import torch


def get_model(MODEL_TYPE):
    # Download MegaDescriptor-T backbone from HuggingFace Hub
    if MODEL_TYPE == 'megadescriptor':
        backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
        embedding_size = 768
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 224
    elif MODEL_TYPE == 'lynx_megadescriptorV3':
        backbone = timm.create_model('hf-hub:strakajk/LynxV3-MegaDescriptor-T-224', num_classes=0, pretrained=True)
        embedding_size = 768
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 224
    elif MODEL_TYPE == 'lynx_megadescriptorV4':
        backbone = timm.create_model('hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256', num_classes=0, pretrained=True)
        embedding_size = 768
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 256
    elif MODEL_TYPE == 'miewid':
        backbone = AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)
        embedding_size = 2152
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img_size = 440
    elif MODEL_TYPE == 'dinov2':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        embedding_size = 384
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img_size = 224
    elif MODEL_TYPE == 'dinov3':
        REPO_DIR = '/home/kargin/Projects/repositories/dinov3'
        backbone = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='/shared/results/common/kargin/unreal_engine/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        embedding_size = 384
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img_size = 224
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported")
    return backbone, embedding_size, mean, std, img_size