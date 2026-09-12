"""The brain: sliding window + PLiX per window + threshold + merge."""

from dataclasses import dataclass

from . import audio as audio_mod
from . import model as model_mod


@dataclass
class Detection:
    keyword: str
    start: float
    end: float
    score: float


def discover_keyword_examples(examples_dir: str) -> dict[str, list[str]]:
    """examples/keyword1/*.wav, examples/keyword2/*.wav -> {"keyword1": [...], ...}"""
    import os

    keyword_paths: dict[str, list[str]] = {}
    for keyword in sorted(os.listdir(examples_dir)):
        kw_dir = os.path.join(examples_dir, keyword)
        if not os.path.isdir(kw_dir):
            continue
        wavs = [
            os.path.join(kw_dir, f)
            for f in sorted(os.listdir(kw_dir))
            if f.lower().endswith(".wav")
        ]
        if wavs:
            keyword_paths[keyword] = wavs
    return keyword_paths


def detect(
    examples_dir: str,
    query_path: str,
    encoder_name: str = "base",
    language: str = "multi",
    device: str = "cpu",
    threshold: float = 0.5,
    hop_seconds: float = 0.25,
) -> list[Detection]:
    """Slide a window across query_path and return windows above threshold."""
    keyword_paths = discover_keyword_examples(examples_dir)
    if not keyword_paths:
        raise ValueError(f"No keyword example folders with .wav files found in {examples_dir}")

    fws_model = model_mod.load_plix(encoder_name=encoder_name, language=language, device=device)
    support = model_mod.build_support_set(keyword_paths, device=device)

    query_waveform = audio_mod.preprocess(query_path)
    windows = audio_mod.make_windows(query_waveform, hop_seconds=hop_seconds)

    detections: list[Detection] = []
    for window in windows:
        query = model_mod.build_query_from_tensor(window["audio"], device=device)
        result = model_mod.predict(fws_model, support, query)

        if result["scores"] is not None:
            score = result["scores"][result["label_index"]]
        else:
            score = 1.0  # no real score available yet, see model.py

        if score >= threshold:
            detections.append(Detection(
                keyword=result["label"],
                start=round(window["start"], 3),
                end=round(window["end"], 3),
                score=round(score, 3),
            ))

    return merge_adjacent(detections)


def merge_adjacent(detections: list[Detection], gap_tolerance: float = 0.3) -> list[Detection]:
    """Collapse consecutive windows of the same keyword into one span."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: (d.keyword, d.start))
    merged: list[Detection] = [detections[0]]

    for det in detections[1:]:
        last = merged[-1]
        same_keyword = det.keyword == last.keyword
        close_enough = det.start <= last.end + gap_tolerance
        if same_keyword and close_enough:
            merged[-1] = Detection(
                keyword=last.keyword,
                start=last.start,
                end=max(last.end, det.end),
                score=max(last.score, det.score),
            )
        else:
            merged.append(det)

    return merged