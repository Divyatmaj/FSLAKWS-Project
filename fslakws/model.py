
"""Thin wrapper around plixkws: load model, build support/query, predict."""

import torch

from plixkws import model as plix_model
from plixkws import util as plix_util


def load_plix(
    encoder_name: str = "base",
    language: str = "multi",
    device: str = "cpu",
):
    """Load a pretrained PLiX model."""
    return plix_model.load(
        encoder_name=encoder_name,
        language=language,
        device=device,
    )


def build_support_set(
    keyword_paths: dict[str, list[str]],
    device: str = "cpu",
) -> dict:
    """keyword_paths: {"hello": ["ex1.wav", ...], ...} -> PLiX support dict."""
    classes = list(keyword_paths.keys())

    all_paths = []
    all_labels = []

    for idx, kw in enumerate(classes):
        for path in keyword_paths[kw]:
            all_paths.append(path)
            all_labels.append(idx)

    support = {
        "paths": all_paths,
        "classes": classes,
        "labels": torch.tensor(all_labels),
    }

    support["audio"] = torch.stack(
        [plix_util.load_clip(path) for path in support["paths"]]
    )

    return plix_util.batch_device(support, device=device)


def build_query_from_tensor(
    audio_chunk: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """Build a PLiX query from a preprocessed 1-second waveform."""
    audio_batched = audio_chunk.unsqueeze(0).unsqueeze(0)
    query = {"audio": audio_batched}

    return plix_util.batch_device(query, device=device)


def compute_prototypes(
    fws_model,
    support: dict,
) -> torch.Tensor:
    """Embed support examples and average embeddings per class."""
    with torch.no_grad():
        embeddings = fws_model.backbone(support["audio"])

    grouped = []

    for idx in range(len(support["classes"])):
        grouped.append(embeddings[support["labels"] == idx])

    grouped = torch.stack(grouped)

    return grouped.mean(dim=1)


def predict_from_prototypes(
    fws_model,
    prototypes: torch.Tensor,
    query: dict,
    classes: list[str],
) -> dict:
    """Predict a query using class prototypes and return confidence scores."""
    with torch.no_grad():
        query_embeddings = fws_model.backbone(query["audio"])

        distances = torch.cdist(
            query_embeddings.unsqueeze(0),
            prototypes.unsqueeze(0),
            p=2,
        ).squeeze(0)

        logits = -(distances ** 2)
        probs = torch.softmax(logits, dim=1)

    label_index = int(torch.argmax(probs, dim=1)[0].item())

    return {
        "label_index": label_index,
        "label": classes[label_index],
        "scores": probs[0].tolist(),
    }

