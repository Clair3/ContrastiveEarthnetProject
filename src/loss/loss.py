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


def contrastive_loss(veg_emb, weather_emb, negative_embs, temperature=0.07):
    """
    Args:
        veg_emb: [B, d] anchor vegetation embedding
        weather_emb: [B, d] positive weather embedding
        negative_embs: list of [B, d] embeddings (either veg or weather negatives)
    """
    B, d = veg_emb.shape
    print(veg_emb.shape, weather_emb.shape)
    print(len(negative_embs), negative_embs[0].shape)

    # Normalize embeddings
    veg_emb = F.normalize(veg_emb, dim=-1)
    weather_emb = F.normalize(weather_emb, dim=-1)
    negative_embs = [F.normalize(n, dim=-1) for n in negative_embs]

    # Concatenate positive + negatives for each anchor
    candidates = torch.cat([weather_emb] + negative_embs, dim=0)  # [B*(1+n), d]

    # Compute similarity: veg → all candidates
    logits = torch.matmul(veg_emb, candidates.T) / temperature  # [B, B*(1+n)]

    # Positive is first element of each row
    labels = torch.arange(B, device=veg_emb.device)

    # Cross-entropy over 1 positive + n negatives
    loss = F.cross_entropy(logits, labels)
    return loss


def contrastive_loss_debug(veg_emb, weather_emb, negative_embs, temperature=0.07):
    B, _ = veg_emb.shape

    # Normalize embeddings
    veg_emb = F.normalize(veg_emb, dim=-1)
    weather_emb = F.normalize(weather_emb, dim=-1)

    # Compute similarity: veg → all candidates
    logits = torch.matmul(veg_emb, weather_emb.T) / temperature  # [B, B*(1+n)]

    # Positive is first element of each row
    labels = torch.arange(B, device=veg_emb.device)

    # Cross-entropy over 1 positive + n negatives
    loss = F.cross_entropy(logits, labels)
    return loss
