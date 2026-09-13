
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
    iou_threshold: float = 0.3,
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

    return non_max_suppress(detections, iou_threshold=iou_threshold)


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


def _iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two detections' time spans."""
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - intersection

    return intersection / union if union > 0 else 0.0


def non_max_suppress(
    detections: list[Detection],
    iou_threshold: float = 0.3,
) -> list[Detection]:
    """Greedy NMS per keyword: anchor each cluster on its top-score window."""
    if not detections:
        return []

    by_keyword: dict[str, list[Detection]] = {}

    for det in detections:
        by_keyword.setdefault(det.keyword, []).append(det)

    kept: list[Detection] = []

    for keyword, dets in by_keyword.items():
        remaining = sorted(dets, key=lambda d: d.score, reverse=True)

        while remaining:
            best = remaining.pop(0)

            overlapping = [d for d in remaining if _iou(best, d) > iou_threshold]
            remaining = [d for d in remaining if _iou(best, d) <= iou_threshold]

            span_start = min([best.start] + [d.start for d in overlapping])
            span_end = max([best.end] + [d.end for d in overlapping])

            kept.append(
                Detection(
                    keyword=keyword,
                    start=round(span_start, 3),
                    end=round(span_end, 3),
                    score=best.score,
                )
            )

    return sorted(kept, key=lambda d: (d.keyword, d.start))
