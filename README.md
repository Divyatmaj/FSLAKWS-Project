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

# real detection
python3 -m fslakws.cli detect \
    --examples_dir=examples \
    --query_path=query.wav
```

Useful flags on `detect`:

| Flag | Default | What it does |
|---|---|---|
| `--encoder_name` / `--language` | `base` / `multi` | which PLiX checkpoint to load (see PLiX's model table) |
| `--hop_seconds` | `0.25` | how far the sliding window moves each step (smaller = finer localization, more compute) |
| `--smoothing_radius` | `2` | majority-vote smoothing over this many neighboring windows, to reduce flicker |
| `--device` | `cpu` | try `mps` on Apple Silicon once `cpu` works |

## Known limitations (honest, current state of V1)

- **No real confidence score.** This build of `plixkws` returns only a
  hard predicted class index, not per-class logits/distances — so every
  detection currently prints `score=1.00`. The `threshold` flag is
  effectively inert until we hook into PLiX's internal ProtoNet
  distances directly. This is the main planned V2 improvement.
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

- Real per-class confidence scores by reaching into PLiX's ProtoNet
  distances directly, instead of treating it as a black box.
- Non-max suppression instead of simple adjacent-merge, for cleaner
  overlapping detections.
- Benchmark CPU vs MPS inference latency/throughput on Apple Silicon.
- Multi-keyword support tested end-to-end (currently only tested with
  one keyword + one negative class).
