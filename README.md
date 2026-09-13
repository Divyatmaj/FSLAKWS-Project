# FSLAKWS 
# FSLAKWS — Few-Shot Localized Audio Keyword Spotting

A small system that takes just a handful of example recordings of a spoken
word (e.g. 3-5 clips of you saying "hello"), and then searches a longer
audio recording to find **where** and **whether** that word occurs —
without training any model from scratch.

```
examples/hello/ex1.wav        query.wav (37s recording)
examples/hello/ex2.wav    +   "...blah blah hello blah..."
examples/hello/ex3.wav
         │                         │
         └───────────┬─────────────┘
                      ▼
                 fslakws.cli detect
                      ▼
        Keyword: hello
        Detected:
            12.4 - 13.1 s
            28.7 - 29.4 s
```

## How it works

I don't train anything ourselves. We use [PLiX](https://github.com/FewshotML/plix)
(`plixkws`), a pretrained multilingual few-shot keyword spotting model —
it was trained on 12M+ one-second clips across 20 languages, and can
recognize a brand-new word from as few as 1-5 support examples.

PLiX itself only classifies fixed ~1-second clips (support examples vs
one query clip). It does **not** localize a keyword inside a long
recording on its own. So this project builds the missing piece around
it:

```
long audio
    │
    ▼
sliding 1-second window (with overlap, so we don't miss a word
    │                     straddling a window boundary)
    ▼
PLiX classifies each window against your keyword(s)
    │
    ▼
majority-vote smoothing across neighboring windows
    │                     (reduces flicker between near-identical,
    │                      heavily overlapping windows)
    ▼
drop windows below --threshold confidence, and any window
predicted as a background/negative class
    ▼
merge adjacent same-label windows into a single detection span
```

## Project structure

```
fslakws/
├── fslakws/
│   ├── __init__.py
│   ├── audio.py       # load/mono/resample/normalize + sliding windows
│   ├── model.py       # thin wrapper around plixkws (load model, predict)
│   ├── detector.py    # the brain: windowing + smoothing + merging
│   └── cli.py         # python-fire CLI entrypoint
├── examples/
│   ├── hello/          # 3+ example clips of your keyword
│   │   ├── ex1.wav
│   │   └── ...
│   └── negative/       # 3+ clips that are NOT the keyword (same count as above)
│       └── ...
├── smoke_test.py       # verifies your environment/install works at all
├── requirements.txt
├── environment.yml
├── pyproject.toml      # registers the `fslakws` console-script entry point
└── .gitignore
```

**Important:** every folder under `examples/` needs the **same number**
of `.wav` files (PLiX requires equal shots per class — see Debugging
below). You also need at least 2 classes: your real keyword, plus a
`negative`/`background` folder of non-keyword audio, or PLiX has nothing
to contrast against and will "detect" the keyword everywhere.

## Setup

```bash
git clone <your-repo-url>
cd fslakws

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate fslakws

# optional: registers the `fslakws` command so you can drop the
# `python -m` prefix everywhere below
pip install -e .
```

## Verify your install

Run this before anything else — it checks imports, downloads/loads the
PLiX weights, and runs a dummy inference call:

```bash
python3 smoke_test.py
```

You should see `All checks passed. Environment is ready for real audio.`
at the end. If it fails partway, the printed step tells you exactly
where — see Debugging below for the common ones.

## Usage

```bash
# quick sanity check (torch/MPS availability)
python3 -m fslakws.cli info

# zero flags -- uses examples/ and query.wav in the current directory
python3 -m fslakws.cli detect

# real detection, folder-based, explicit paths
python3 -m fslakws.cli detect \
    --examples_dir=examples \
    --query_path=query.wav

# real detection, ad-hoc (no folder needed) -- pass each keyword as its
# own flag with comma-separated .wav paths
python3 -m fslakws.cli detect \
    --query_path=query.wav \
    --hello=ex1.wav,ex2.wav,ex3.wav \
    --negative=neg1.wav,neg2.wav,neg3.wav
```

If you ran `pip install -e .`, drop the `python -m fslakws.cli` prefix
and just use `fslakws` (e.g. `fslakws detect`, `fslakws info`).

Output is a color-coded table per keyword (score: green ≥0.8, yellow
0.5-0.8, red <0.5) instead of plain text.

Useful flags on `detect`:

| Flag | Default | What it does |
|---|---|---|
| `--query_path` | `query.wav` | the long recording to search |
| `--examples_dir` | `examples` | folder-based support set (see Project structure) — omit this if using ad-hoc `--keyword=...` flags instead |
| `--encoder_name` / `--language` | `base` / `multi` | which PLiX checkpoint to load (see PLiX's model table) |
| `--threshold` | `0.5` | minimum confidence (0-1) a window's top class must reach to be reported |
| `--hop_seconds` | `0.25` | how far the sliding window moves each step (smaller = finer localization, more compute) |
| `--smoothing_radius` | `2` | majority-vote smoothing over this many neighboring windows, to reduce flicker |
| `--batch_size` | `8` | how many sliding windows to embed per backbone forward pass |
| `--device` | `cpu` | try `mps` on Apple Silicon once `cpu` works |

## Fixed since V1

- **Real per-class confidence scores.** `plixkws`'s own `ProtoNet.forward()`
  (see its installed source) computes squared-distance-based logits
  internally but only ever returns `torch.argmax(...)` — the confidence
  never left the library, which is why V1 had to hardcode `score=1.00` on
  every detection. Fixed by calling `fws_model.backbone` directly (a
  public attribute) in `model.py` and reproducing PLiX's own
  prototype-distance math ourselves, but keeping the softmax probabilities
  instead of discarding them (`compute_prototypes` / `predict_from_prototypes`).
  As a side effect, the support set is now embedded once per `detect()`
  run instead of being re-embedded on every single sliding window.
- **`--threshold` now actually filters.** It was accepted as a CLI/detector
  parameter but never referenced anywhere inside `detect()` — dead code.
  Now that scores are real, a window is only reported if its confidence
  clears `--threshold`.

We also tried swapping PLiX's backbone for general-purpose pretrained
audio embeddings (LAION CLAP, WavLM mean-pooled) hoping for stronger
separation between keywords. Tested directly against the real clips in
`examples/` and `query.wav`: both scored almost every window close to
50/50 and separated "hello" from "negative" worse than PLiX (prototype
distance ~0.8-1.0 vs. PLiX's confident near-1.0 predictions). PLiX's
backbone was trained end-to-end with an episodic few-shot loss
specifically to make the prototype-then-distance scheme work; generic
pretrained embeddings aren't optimized for that geometry. Reverted — PLiX
stays the backbone.

- **Query windows are now embedded in batches instead of one at a time.**
  `detect()` used to call `fws_model.backbone(...)` separately for every
  single sliding window (~100+ calls for a 37s clip) — the same
  one-at-a-time pattern the support set used to have before it was fixed
  to embed once per run. Now windows are grouped into fixed-size batches
  (`--batch_size`, default 8) and each batch runs through the backbone in
  one forward pass (`model.build_query_batch` / `predict_batch_from_prototypes`),
  cutting real end-to-end wall time roughly in half on our test clip on CPU.
  We first tried embedding *all* windows in one giant batch (no chunking at
  all) expecting an even bigger win — measured instead, on this CPU, it was
  slower than the original one-at-a-time loop, and past a certain batch size
  (16 on our machine) it fell off a cliff to ~3x slower than the loop, likely
  a threading/cache effect inside PyTorch's CPU conv kernels at larger batch
  sizes. `--batch_size` is exposed as a flag precisely because this sweet
  spot is hardware/library-dependent — 8 is a safe default measured on this
  machine, not a universal optimum.

## CLI UX improvements

- **Colorful output.** `detect`/`info` now print via `rich` instead of
  plain `print()` — a color-coded table per keyword (green/yellow/red by
  score) instead of raw text lines.
- **Ad-hoc audio paths.** You no longer have to pre-organize files into
  `examples/<keyword>/`. Pass `--<keyword>=path1,path2,...` flags directly
  and `detect` builds the same support set internally
  (`detector.detect()` now accepts a `keyword_paths` dict directly, not
  just `examples_dir`). Folder-based mode still works unchanged.
- **Shorter command.** `query_path`/`examples_dir` default to `query.wav`/
  `examples`, so `python -m fslakws.cli detect` works with zero flags.
  `pyproject.toml` also registers a `fslakws` console-script entry point
  (`pip install -e .`) so you can drop the `python -m fslakws.cli` prefix
  entirely — e.g. `fslakws detect`.

## Known limitations (honest, current state)

- **Few-shot classification on short, heavily-overlapping windows is
  noisy.** With only 3-5 examples per class, predictions can flicker
  between windows that are 75%+ identical. `smoothing_radius` helps but
  doesn't eliminate this.
- **Negative examples must be domain-matched.** Generic stock
  sound-effect clips as your "negative" class tend to make the model
  separate "sound-effect vs voice recording," not "keyword vs no
  keyword." Best results come from negative clips recorded in the same
  environment/mic as your keyword clips (or literally cropped from your
  own query recording, from stretches where the keyword isn't said).

## Debugging

**`ModuleNotFoundError: No module named 'fslakws'`**
You're running the command from the wrong directory. `cd` to the folder
that directly *contains* the `fslakws/` package folder (same level as
`examples/`, `smoke_test.py`), not from inside it.

**`SSL: CERTIFICATE_VERIFY_FAILED` (during model download)**
Common on macOS with the python.org installer — Python's own cert store
isn't hooked up. Fix:
```bash
# Finder → Applications → Python 3.x → double-click "Install Certificates.command"
# or, from terminal:
pip install --upgrade certifi
```
If that doesn't fix it, you may be on a network doing SSL inspection
(school/corporate wifi, VPN) — try a different network once to confirm.

**`FAILED to load model: 'base_multi_multi'` (or similar double-suffix name)**
Wrong `encoder_name`/`language` combo. PLiX builds the checkpoint name
internally as `{encoder_name}_{language}`. Use `encoder_name="base"`,
`language="multi"` for the multilingual model — don't pass the already-
combined name (like `"base_multi"`) as `encoder_name` itself.

**`RuntimeError: Could not load libtorchcodec` / FFmpeg errors**
`torchaudio.load()` in newer versions needs FFmpeg via TorchCodec, which
can fail to link properly on macOS. This project avoids the issue by
loading `.wav` files with `soundfile` instead (see `audio.py`) — make
sure `soundfile` is installed (`pip install soundfile`, already in
`requirements.txt`).

**`RuntimeError: stack expects each tensor to be equal size, but got [3, 1280] ... [5, 1280]`**
Your keyword folders have different numbers of example clips. PLiX
requires the same number of support examples (K) per class — make sure
e.g. `examples/hello/` and `examples/negative/` both have exactly the
same number of `.wav` files.

**`command not found: python`**
Use `python3` — modern macOS doesn't alias `python` by default.

**Squiggly underlines under imports in VS Code, even after `pip install`**
VS Code is pointed at a different Python interpreter than the one you
installed packages into. `Cmd+Shift+P` → `Python: Select Interpreter` →
pick the same one your terminal uses (check with `which python3`), then
`Cmd+Shift+P` → `Developer: Reload Window`.

## Roadmap (post-V1)

- Non-max suppression instead of simple adjacent-merge, for cleaner
  overlapping detections.
- Benchmark CPU vs MPS inference latency/throughput on Apple Silicon.
- Multi-keyword support tested end-to-end (currently only tested with
  one keyword + one negative class).
