"""Thin wrapper around plixkws: load model, build support/query, predict."""

import torch
from plixkws import model as plix_model
from plixkws import util as plix_util


def load_plix(encoder_name: str = "base", language: str = "multi", device: str = "cpu"):
    """Load a pretrained PLiX model. Checkpoint name = "{encoder_name}_{language}"."""
    return plix_model.load(encoder_name=encoder_name, language=language, device=device)


def build_support_set(keyword_paths: dict[str, list[str]], device: str = "cpu") -> dict:
    """keyword_paths: {"hello": ["ex1.wav", ...], ...} -> plixkws support dict."""
    classes = list(keyword_paths.keys())
    all_paths = []
    all_labels = []
    for idx, kw in enumerate(classes):
        for p in keyword_paths[kw]:
            all_paths.append(p)
            all_labels.append(idx)

    support = {
        "paths": all_paths,
        "classes": classes,
        "labels": torch.tensor(all_labels),
    }
    support["audio"] = torch.stack([plix_util.load_clip(p) for p in support["paths"]])
    support = plix_util.batch_device(support, device=device)
    return support


def build_query_from_tensor(audio_chunk: torch.Tensor, device: str = "cpu") -> dict:
    """Build a plixkws query dict from a preprocessed 1-second waveform tensor."""
    audio_batched = audio_chunk.unsqueeze(0).unsqueeze(0)
    query = {"audio": audio_batched}
    return plix_util.batch_device(query, device=device)


def predict(fws_model, support: dict, query: dict) -> dict:
    """Run inference. Returns {"label_index", "label", "scores" (or None)}."""
    with torch.no_grad():
        raw = fws_model(support, query)

    scores = None
    if raw.dim() >= 2 and raw.shape[-1] == len(support["classes"]):
        probs = torch.softmax(raw, dim=-1)
        label_index = int(torch.argmax(probs, dim=-1)[0].item())
        scores = probs[0].tolist()
    else:
        label_index = int(raw[0].item()) if raw.dim() > 0 else int(raw.item())

    label = support["classes"][label_index]
    return {"label_index": label_index, "label": label, "scores": scores}