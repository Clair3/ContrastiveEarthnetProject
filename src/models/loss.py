import torch
import torch.nn.functional as F


def info_nce_loss(veg_emb, weather_emb, temperature=0.07):
    """
    Args:
        veg_emb: [B, d]
        weather_emb: [B, d]
    """
    # Normalize embeddings
    veg_emb = F.normalize(veg_emb, dim=-1)
    weather_emb = F.normalize(weather_emb, dim=-1)

    # Similarity matrix between all pairs
    logits = torch.matmul(veg_emb, weather_emb.T) / temperature  # [B, B]
    labels = torch.arange(len(veg_emb), device=veg_emb.device)

    # Cross-entropy over similarities
    loss_v2w = F.cross_entropy(logits, labels)
    loss_w2v = F.cross_entropy(logits.T, labels)
    return (loss_v2w + loss_w2v) / 2
