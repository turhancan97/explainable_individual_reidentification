import torch.nn as nn
import torch
from pytorch_metric_learning import distances, losses, miners


class ArcFaceLoss(nn.Module):
    """
    Wraps Pytorch Metric Learning ArcFaceLoss.

    Args:
        num_classes (int): Number of classes.
        embedding_size (int): Size of the input embeddings.
        margin (int, optional): Margin for ArcFace loss (in radians).
        scale (int, optional): Scale parameter for ArcFace loss.
    """

    def __init__(self, num_classes: int, embedding_size: int, margin: int = 0.5, scale: int = 64):

        super().__init__()
        self.loss = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=57.3 * margin,
            scale=scale,
        )

    def forward(self, embeddings, y):
        return self.loss(embeddings, y)


class TripletLoss(nn.Module):
    """
    Wraps Pytorch Metric Learning TripletMarginLoss.

    Args:
        margin (int, optional): Margin for triplet loss.
        mining (str, optional): Type of triplet mining. One of: 'all', 'hard', 'semihard'
        distance (str, optional): Distance metric for triplet loss. One of: 'cosine', 'l2', 'l2_squared'

    """

    def __init__(self, margin: int = 0.2, mining: str = "semihard", distance: str = "l2_squared"):

        super().__init__()
        if distance == "cosine":
            distance = distances.CosineSimilarity()
        elif distance == "l2":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        elif distance == "l2_squared":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=2)
        else:
            raise ValueError(f"Invalid distance: {distance}")

        self.loss = losses.TripletMarginLoss(distance=distance, margin=margin)
        self.miner = miners.TripletMarginMiner(distance=distance, type_of_triplets=mining, margin=margin)

    def forward(self, embeddings, y):
        indices_tuple = self.miner(embeddings, y)
        return self.loss(embeddings, y, indices_tuple)


class SoftmaxLoss(nn.Module):
    """
    CE with single dense layer classification head.

    Args:
        num_classes (int): Number of classes.
        embedding_size (int): Size of the input embeddings.
    """

    def __init__(self, num_classes: int, embedding_size: int):

        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(embedding_size, num_classes)

    def forward(self, x, y):
        return self.criterion(self.linear(x), y)

    def predict_probabilities(self, x):
        logits = self.linear(x)
        return logits.softmax(dim=1)


class SoftmaxLossEP(nn.Module):
    """
    Efficient probing head operating on ViT patch-token embeddings.

    Input shape is expected as (B, N, C), where N is the number of patch tokens.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        dropout_rate: float = 0.0,
        num_queries: int = 4,
        d_out: int = 2,
    ):
        super().__init__()
        if d_out <= 0:
            raise ValueError("d_out must be > 0")
        if num_queries <= 0:
            raise ValueError("num_queries must be > 0")
        if embedding_size % d_out != 0:
            raise ValueError(f"embedding_size ({embedding_size}) must be divisible by d_out ({d_out})")
        if embedding_size % (d_out * num_queries) != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by d_out*num_queries ({d_out * num_queries})"
            )

        self.scale = embedding_size**-0.5
        self.num_heads = 1
        self.d_out = d_out
        self.num_queries = num_queries
        self.v = nn.Linear(embedding_size, embedding_size // d_out, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, embedding_size) * 0.02)
        self.layer_norm = nn.LayerNorm(embedding_size // d_out)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(embedding_size // d_out, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SoftmaxLossEP expects input shape (B, N, C). Got {tuple(x.shape)}")
        bsz, num_tokens, emb = x.shape
        cls_token = self.cls_token.expand(bsz, -1, -1)
        q = cls_token.reshape(bsz, self.num_queries, self.num_heads, emb // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(bsz, num_tokens, self.num_heads, emb // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(bsz, num_tokens, self.num_queries, emb // (self.d_out * self.num_queries)).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attention_map = attn.squeeze(1)
        self.attention_map = attention_map
        pooled = torch.matmul(attention_map.unsqueeze(2), v).view(bsz, emb // self.d_out)
        return pooled

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled(x)
        pooled = self.layer_norm(pooled)
        pooled = self.proj_drop(pooled)
        return self.linear(pooled)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(self._logits(x), y)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._logits(x), dim=1)
