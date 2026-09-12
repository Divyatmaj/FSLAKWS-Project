
"""CLI interface for FSLAKWS using Python Fire."""

import fire

from . import detector


class FSLAKWS:

    def detect(
        self,
        examples_dir: str,
        query_path: str,
        encoder_name: str = "base",
        language: str = "multi",
        device: str = "cpu",
        threshold: float = 0.5,
        hop_seconds: float = 0.25,
        smoothing_radius: int = 2,
    ):
        """Detect keywords in a query audio file."""
        detections = detector.detect(
            examples_dir=examples_dir,
            query_path=query_path,
            encoder_name=encoder_name,
            language=language,
            device=device,
            threshold=threshold,
            hop_seconds=hop_seconds,
            smoothing_radius=smoothing_radius,
        )

        if not detections:
            print("No detections above threshold.")
            return

        by_keyword: dict[str, list[detector.Detection]] = {}

        for detection in detections:
            by_keyword.setdefault(detection.keyword, []).append(detection)

        for keyword, hits in by_keyword.items():
            print(f"\nKeyword: {keyword}")
            print("Detected:")

            for hit in hits:
                print(
                    f"    {hit.start:.1f} - {hit.end:.1f} s   "
                    f"score={hit.score:.2f}"
                )

    def info(self):
        """Show package and device information."""
        import torch

        print("fslakws V1")
        print(f"torch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")


def main():
    fire.Fire(FSLAKWS)


if __name__ == "__main__":
    main()

