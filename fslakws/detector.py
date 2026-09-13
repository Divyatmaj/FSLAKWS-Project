
"""Detect keywords in long audio using sliding-window PLiX inference."""

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
    """Find .wav examples grouped by keyword folder."""
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
    query_path: str,
    examples_dir: str | None = None,
    keyword_paths: dict[str, list[str]] | None = None,
    encoder_name: str = "base",
    language: str = "multi",
    device: str = "cpu",
    threshold: float = 0.5,
    hop_seconds: float = 0.25,
    smoothing_radius: int = 2,
    batch_size: int = 8,
    background_labels: tuple[str, ...] = ("negative", "background"),
) -> list[Detection]:
    """
    Run sliding-window keyword detection over a query audio file.

    Provide either examples_dir or keyword_paths.
    keyword_paths takes priority if both are provided.
    """
    if keyword_paths is None:
        if examples_dir is None:
            raise ValueError("Provide either examples_dir or keyword_paths")

        keyword_paths = discover_keyword_examples(examples_dir)

    if not keyword_paths:
        raise ValueError(
            f"No keyword example folders with .wav files found in {examples_dir}"
        )

    fws_model = model_mod.load_plix(
        encoder_name=encoder_name,
        language=language,
        device=device,
    )

    support = model_mod.build_support_set(
        keyword_paths,
        device=device,
    )

    prototypes = model_mod.compute_prototypes(
        fws_model,
        support,
    )

    query_waveform = audio_mod.preprocess(query_path)

    windows = audio_mod.make_windows(
        query_waveform,
        hop_seconds=hop_seconds,
    )

    raw_results: list[dict] = []

    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]

        query_batch = model_mod.build_query_batch(
            [window["audio"] for window in chunk],
            device=device,
        )

        raw_results.extend(
            model_mod.predict_batch_from_prototypes(
                fws_model,
                prototypes,
                query_batch,
                support["classes"],
            )
        )

    raw_labels = [result["label"] for result in raw_results]

    smoothed_labels = smooth_labels(
        raw_labels,
        radius=smoothing_radius,
    )

    detections: list[Detection] = []

    for window, label, result in zip(
        windows,
        smoothed_labels,
        raw_results,
    ):
        if label in background_labels:
            continue

        score = result["scores"][support["classes"].index(label)]

        if score < threshold:
            continue

        detections.append(
            Detection(
                keyword=label,
                start=round(window["start"], 3),
                end=round(window["end"], 3),
                score=round(score, 4),
            )
        )

    return merge_adjacent(detections)


def smooth_labels(labels: list[str], radius: int) -> list[str]:
    """Apply majority-vote smoothing over neighboring windows."""
    if radius <= 0 or not labels:
        return labels

    import collections

    smoothed = []
    n = len(labels)

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)

        neighborhood = labels[lo:hi]
        counts = collections.Counter(neighborhood)
        top_label, top_count = counts.most_common(1)[0]

        # Keep the original label when tied.
        if list(counts.values()).count(top_count) > 1:
            smoothed.append(labels[i])
        else:
            smoothed.append(top_label)

    return smoothed


def merge_adjacent(
    detections: list[Detection],
    gap_tolerance: float = 0.3,
) -> list[Detection]:
    """Merge overlapping or nearby detections of the same keyword."""
    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: (d.keyword, d.start),
    )

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
