# Reproducing v1.0

How to install the code, get the data and the v1.0 checkpoints, regenerate the published files,
rerun the experiments and compare a rerun with the committed results. The results themselves are
summarized in the [README](../README.md) and answered in [questions.md section
C](questions.md#c-the-v10-answers).

## Installation

Tested on Python 3.12 (3.12.8 on macOS, Apple silicon; 3.12 on Linux x86_64 in continuous
integration); `requires-python` is `>=3.12`. Run every command from the repository root: the
default paths of the data and the results are relative to the working directory (`data/`,
`results/`), and the tools are run as modules of the `tools` package of the repository root
(`python -m tools.<name>`).

macOS on Apple silicon (MPS):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

Linux, CPU only, with the CPU-only PyTorch wheel installed first:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
pip install -r requirements.txt
pip install --no-deps -e .
```

`requirements.txt` pins every package, including the dependencies of the dependencies. Both
blocks install the package `ebt` (`src/ebt`) in editable mode (`-e`). `pyproject.toml` lists
torch, numpy and matplotlib at the pinned versions, and `pip install -e '.[dev]'` adds pytest and
ruff.

Use the editable install, or put the source on the path with `export PYTHONPATH=src:.`, for every
command that reads or writes checkpoints. `src/ebt/checkpoints.py` finds the repository root from
its own location. With a non-editable install (`pip install .`) that location is inside the
environment, so the checkpoints would be read from and written to a directory there, not to
`checkpoints/` of the repository. The tests are not affected: pytest puts `src` first on the path
(`pythonpath` in `pyproject.toml`), so they import the repository's source even when continuous
integration installs the package without `-e`. Devices are `--device mps` and `--device cpu` only;
there is no CUDA path, and on Linux every command uses `--device cpu`.

Package name and version. The distribution is named `equipoise-lm`, and the import package keeps
the name `ebt`, the EBT implementation it started as. Renaming it would change its source files and
with them the code fingerprints of every result. For the same reason the code version stays 0.1.0:
`src/ebt/__init__.py`, which holds `__version__`, is hashed into every code fingerprint, so
changing it would make every v1.0 result stale. Release v1.0 is code version 0.1.0, the version
recorded in the `environment` field of every result file.

## Data

The data are not in the repository and are not redistributed. Each file is checked by its SHA-256
before use; a file with another checksum is refused.

| File | Expected path | Source | Licence or terms |
|---|---|---|---|
| MNIST training images | `data/mnist/raw/train-images-idx3-ubyte.gz` | <https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz> (the mirror listed by torchvision) | the source page, Y. LeCun, C. Cortes and C. J. C. Burges, <https://yann.lecun.com/exdb/mnist>, states no licence; the Hugging Face dataset card lists MIT (<https://huggingface.co/datasets/ylecun/mnist>); downloaded from the PyTorch mirror in the Source column; not redistributed here |
| MNIST test images | `data/mnist/raw/t10k-images-idx3-ubyte.gz` | <https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz> | as above |
| Tiny Shakespeare | `data/text/raw/tinyshakespeare.txt` | <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>; the file is `input.txt` in its source and is saved under the expected name | Shakespeare's text is in the public domain; the char-rnn repository that distributes the file has no licence file ([Tiny Shakespeare], method.md section 18.1) |
| Pride and Prejudice (out-of-distribution text) | `data/text/raw/pg1342.txt` | <https://gutenberg.pglaf.org/cache/epub/1342/pg1342.txt> | public domain in the USA; use of the file is subject to the Project Gutenberg terms (<https://www.gutenberg.org/policy/license.html>) |
| text8 (v1.1, planned) | not used in v1.0 | <http://mattmahoney.net/dc/textdata.html> ([design.md, References](design.md#references)) | the text is derived from English Wikipedia, whose text is licensed under CC BY-SA and the GFDL |

```sh
mkdir -p data/mnist/raw data/text/raw
curl -L -o data/mnist/raw/train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -L -o data/mnist/raw/t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
curl -L -o data/text/raw/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -L -o data/text/raw/pg1342.txt https://gutenberg.pglaf.org/cache/epub/1342/pg1342.txt
```

The SHA-256 values of the MNIST files are in `src/ebt/tasks/mnist.py` (`SHA256`); those of the two
text files are in `src/ebt/tasks/text.py` and are listed in method.md section 18.1. The files from
these four sources matched them on 23 November 2026. The Gutenberg file used in v1.0 was
downloaded on 20 November 2026; if the source changes the file, its checksum differs and the text
task refuses it. The regression task generates its data and needs no file.

## The v1.0 checkpoints

The checkpoints are not committed. They are published as three archives with the GitHub release
v1.0, one per task, each holding `checkpoints/<experiment>/full/` with its repository-relative
path, and a file `SHA256SUMS`:

| Archive | Contents | Size |
|---|---|---|
| `checkpoints-regression-g2-full.tar.gz` | `checkpoints/regression_g2/full/`, 40 checkpoints | 716 733 bytes |
| `checkpoints-mnist-denoising-full.tar.gz` | `checkpoints/mnist_denoising/full/`, 10 checkpoints | 30 546 731 bytes |
| `checkpoints-text-lm-full.tar.gz` | `checkpoints/text_lm/full/`, 10 checkpoints | 234 851 899 bytes |

With the GitHub command-line tool, in the repository root:

```sh
gh release download v1.0 --repo DeOcxan/equipoise-lm
shasum -a 256 -c SHA256SUMS
for f in checkpoints-*-full.tar.gz; do tar -xzf "$f"; done
```

Without it, with curl, and with `sha256sum` on Linux:

```sh
base=https://github.com/DeOcxan/equipoise-lm/releases/download/v1.0
for f in SHA256SUMS checkpoints-regression-g2-full.tar.gz checkpoints-mnist-denoising-full.tar.gz checkpoints-text-lm-full.tar.gz; do
  curl -LO "$base/$f"
done
sha256sum -c SHA256SUMS
for f in checkpoints-*-full.tar.gz; do tar -xzf "$f"; done
```

Then check every unpacked checkpoint against the SHA-256 recorded in the result files:

```sh
python3 - <<'EOF'
import hashlib, json, pathlib
recorded = {}
for f in pathlib.Path("results").rglob("*.json"):
    c = json.loads(f.read_text()).get("checkpoint")
    if isinstance(c, dict) and "sha256" in c:
        recorded.setdefault(c["path"], set()).add(c["sha256"])
bad = [p for p, s in recorded.items()
       if not pathlib.Path(p).exists() or {hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()} != s]
print(f"{len(recorded) - len(bad)} of {len(recorded)} checkpoints match the result files")
for p in bad: print("MISMATCH or missing:", p)
raise SystemExit(1 if bad else 0)
EOF
```

It prints "45 of 45 checkpoints match the result files". The SHA-256 of a checkpoint is recorded
in the G3/G4 and G5 result files (`checkpoint.sha256`), so it covers the 45 checkpoints that G3,
G4 or G5 evaluate. The 15 baseline checkpoints (`regressor-seed*.pt`, `denoiser-seed*.pt`,
`transformer-seed*.pt`) are not evaluated there. Their training result files record only their
path (`code.checkpoint`), and only `SHA256SUMS` covers them.

## Every published file and how it is made

Every file under `results/` comes from one of the commands below. Only the report tools work from
the committed result files alone. The experiments themselves need the data, and the inference
experiments also need the checkpoints.

| Published files | Command | Needs checkpoints and data |
|---|---|---|
| `results/regression/g2/full/`: `table.md`, `predictions.png`, `energy_*.png`, `density_mdn.png` | `python -m tools.regression_g2_report results/regression/g2/full` | no |
| `results/mnist/denoising/full/`: `table.md`, `examples.png`, `validation.png` | `python -m tools.mnist_report results/mnist/denoising/full` | no |
| `results/text/lm/full/`: `table.md`, `samples.md`, `validation.png` | `python -m tools.text_report results/text/lm/full` | no |
| `results/<task>/g5/full/`: `g5.json`, `table.md` | `python -m tools.g5_report` | no |
| `results/g3_steps.png` | `python -m tools.g3_figure` | no |
| per-seed results, `search.json` and `search/` of `results/regression/g2/full/`, `results/mnist/denoising/full/` and `results/text/lm/full/` | `python -u -m ebt.experiments.regression_g2`, `mnist_denoising` or `text_lm` ([below](#rerunning-the-final-runs-without-overwriting-v10)) | data (MNIST and text); writes the checkpoints |
| `results/<task>/g3_g4/full/`: per-seed results, `summary.json`, `table.md` | `python -u -m ebt.experiments.inference_g3_g4` ([below](#rerunning-g3-g4-and-g5-from-the-checkpoints)) | checkpoints and data |
| `results/<task>/g5/full/` (MNIST: `noise-<level>/`): per-seed results, `summary.json` | `python -u -m ebt.experiments.inference_g5` ([below](#rerunning-g3-g4-and-g5-from-the-checkpoints)) | checkpoints and data |
| `results/text/measurements-2026-11-20/` | a measurement script that is not in the repository (method.md section 18.8) | an MPS device |
| `results/text/measurements-2026-11-21/budget/` | `python -m tools.text_budget OUTPUT_DIR` (method.md sections 18.7 and 18.8); `section-18.8.md` is regenerated as [below](#regenerating-section-188md) | an MPS device to measure; none to regenerate |
| `results/text/measurements-2026-11-21/mps-gradients/` | `python -m tools.mps_text_gradients measure > mps-diag.jsonl`, then `python -m tools.mps_text_gradients decide mps-diag.jsonl > mps-decision.txt` ([numerics.md](numerics.md#tolerance-policy)) | an MPS device to measure; none to decide |

The regression task generates its data, so its experiments need the checkpoints only. To
regenerate the files of the report tools:

```sh
python -m tools.regression_g2_report results/regression/g2/full
python -m tools.mnist_report results/mnist/denoising/full
python -m tools.text_report results/text/lm/full
python -m tools.g5_report
python -m tools.g3_figure
```

`tests/test_published_reports.py` regenerates every `table.md` and the JSON reports, the
`summary.json` of G3 and G4 and the `g5.json` of G5, from the committed per-seed files into a
temporary directory. It compares each with the committed file, JSON number by number and tables
character by character, on the CPU without checkpoints or data. It does not compare
`samples.md`, the `summary.json` files of G5 or the figures.

### Regenerating section-18.8.md

`section-18.8.md` is regenerated without measuring, by the procedure of [method.md section
18.8](method.md#188-timings-and-the-chosen-setting): copy the folder into a temporary directory,
apply the rule there with `--measure-command false`, and copy `section-18.8.md` back.

```sh
copy=$(mktemp -d)
cp -R results/text/measurements-2026-11-21/budget/. "$copy"
python -m tools.text_budget "$copy" --measure-command false
cp "$copy/section-18.8.md" results/text/measurements-2026-11-21/budget/
```

Do not run the tool on `results/` itself: it would rewrite `memory.json` and add `timing.log`
there.

## How the experiments treat existing files

`src/ebt/experiments/runner.py`:

- `--output DIR` moves only the results. Without it, results go to this repository, in place over
  the v1.0 files: `results/regression/g2/full/`, `results/mnist/denoising/full/` and
  `results/text/lm/full/`. Checkpoints are always written to the repository, with or without
  `--output`: `checkpoints/regression_g2/full/`, `checkpoints/mnist_denoising/full/` and
  `checkpoints/text_lm/full/` (`src/ebt/checkpoints.py`). The training experiments have no
  `--checkpoints` flag in v1.0.
- A search configuration whose file exists is not run again. A final run is skipped only if its
  result file holds the chosen configuration and the current code fingerprint and its checkpoint
  exists with that fingerprint. Otherwise it is trained again and both files are replaced.
- Code fingerprint: every search entry, choice and result file records the SHA-256 of the source
  files that compute it (`src/ebt/fingerprint.py`; method.md section 16.6). An entry or choice
  with another fingerprint or none is stale.
- A stage that would use a stale search entry or choice stops before any work and lists them.
  With `--reuse-stale` it uses them as they are, without recomputing them, and lists them in
  `code.reused_stale_files` of every result file it writes.

Warning: a final or full rerun on a machine that holds the v1.0 checkpoints overwrites them, also
with `--output`. That makes the v1.0 G3, G4 and G5 results stale (they belong to the old
checkpoints) and removes the checkpoints that G3, G4 and G5 use.

## Rerunning the final runs without overwriting v1.0

The v1.0 final runs of all three tasks reused searches written by other code. The regression and
MNIST searches were recorded before code fingerprints existed. The text search was recorded under
the fingerprint f448d4e81408, which differs from the current text code by one line of `generate()`
(method.md section 18.5). Every v1.0 final result file lists them in `code.reused_stale_files`, so
`--reuse-stale` is needed to rerun the final runs from the v1.0 searches.

To rerun them without touching the v1.0 files:

1. Move `checkpoints/` aside, for example to `../equipoise-v1.0-checkpoints/`. Training has no
   `--checkpoints` flag in v1.0 and always writes to `checkpoints/` of the repository.
2. Copy each experiment's `search.json` and `search/` into a new output directory.
3. Run `--stage final --reuse-stale` with `--output` set to that directory, and the report tool on
   it.

```sh
mv checkpoints ../equipoise-v1.0-checkpoints
for e in regression/g2 mnist/denoising text/lm; do
  mkdir -p "../equipoise-rerun/$e/full"
  cp -R "results/$e/full/search.json" "results/$e/full/search" "../equipoise-rerun/$e/full/"
done

python -u -m ebt.experiments.regression_g2 --tier full --device cpu --stage final --reuse-stale --output ../equipoise-rerun/regression/g2/full
python -m tools.regression_g2_report ../equipoise-rerun/regression/g2/full

python -u -m ebt.experiments.mnist_denoising --tier full --device mps --stage final --reuse-stale --output ../equipoise-rerun/mnist/denoising/full
python -m tools.mnist_report ../equipoise-rerun/mnist/denoising/full

python -u -m ebt.experiments.text_lm --tier full --device mps --stage final --reuse-stale --output ../equipoise-rerun/text/lm/full
python -m tools.text_report ../equipoise-rerun/text/lm/full
```

On Linux use `--device cpu` for MNIST and text as well. The new checkpoints are written to
`checkpoints/`. To evaluate them, run G3, G4 and G5 as below; to return to the v1.0 checkpoints,
move the new `checkpoints/` aside and move `../equipoise-v1.0-checkpoints` back.

Without `--output` the commands write the results in place, over the v1.0 files. `--stage search`
or `--stage all` would stop at the stale search entries. With `--reuse-stale` they would also run,
under the current code, the regression configurations that v1.0 recorded only in `search.json`
(models (a), (b), (c1) and (c2)). In a new directory without the copied search, `--stage all` runs
the whole search under the current code and then the final runs with its choice. That is not what
v1.0 ran, and the choice may differ.

## Rerunning G3, G4 and G5 from the checkpoints

Questions G3 and G4 on the final checkpoints, without training (method.md section 19). They read
the checkpoints by default from the three `checkpoints/` directories above (`--checkpoints` sets
another directory). Always write a rerun to a new directory with `--output`, which moves the
results only, so that the committed v1.0 results stay as they are:

```sh
python -u -m ebt.experiments.inference_g3_g4 --task regression --tier full --device cpu --output ../equipoise-rerun/regression/g3_g4/full
python -u -m ebt.experiments.inference_g3_g4 --task mnist --tier full --device mps --output ../equipoise-rerun/mnist/g3_g4/full
python -u -m ebt.experiments.inference_g3_g4 --task text --tier full --device mps --output ../equipoise-rerun/text/g3_g4/full
```

Question G5 (method.md section 20) repeats G3 on out-of-distribution data and compares the
relative improvement from thinking in and out of distribution. Write it to the same new tree, and
run the G5 report on that tree with `--results`:

```sh
python -u -m ebt.experiments.inference_g5 --task regression --tier full --device cpu --output ../equipoise-rerun/regression/g5/full
python -u -m ebt.experiments.inference_g5 --task mnist --tier full --device mps --output ../equipoise-rerun/mnist/g5/full
python -u -m ebt.experiments.inference_g5 --task text --tier full --device mps --output ../equipoise-rerun/text/g5/full
python -m tools.g5_report --results ../equipoise-rerun
```

On Linux use `--device cpu` for MNIST and text. `tools/g5_report.py` reads the in-distribution G3
results from `<results>/<task>/g3_g4/full/` and the out-of-distribution results from
`<results>/<task>/g5/full/` (MNIST: one subdirectory `noise-<level>/` per level). It checks that
both come from the same checkpoint, and it writes `g5.json` and `table.md` into
`<results>/<task>/g5/full/`.

Without `--output` these commands write into `results/`. A result is current only for the
checkpoint it was computed on (path, fingerprint and SHA-256). After a retraining the committed G3,
G4 and G5 results are stale: the command without `--output` stops and lists them, and with
`--reuse-stale` it keeps them unchanged instead of recomputing them.

The `fast` tier runs the same code at a small size and is used by the tests.

## Comparing a rerun with the committed files

Compare each rerun table with its committed counterpart:

```sh
for t in regression mnist text; do
  diff "results/$t/g3_g4/full/table.md" "../equipoise-rerun/$t/g3_g4/full/table.md"
  diff "results/$t/g5/full/table.md" "../equipoise-rerun/$t/g5/full/table.md"
done
diff results/regression/g2/full/table.md ../equipoise-rerun/regression/g2/full/table.md
diff results/mnist/denoising/full/table.md ../equipoise-rerun/mnist/denoising/full/table.md
diff results/text/lm/full/table.md ../equipoise-rerun/text/lm/full/table.md
```

What a rerun can reproduce ([numerics.md, Reproducibility](numerics.md#reproducibility)):

- Regression ran on the CPU. A rerun with the same torch version on the same platform (macOS on
  Apple silicon for v1.0) is expected to give identical files. On another platform or torch
  version the numbers may differ.
- MNIST and text ran on MPS, whose training runs are not bit-reproducible. No exact agreement with
  the committed values is claimed: the MPS tolerances of numerics.md cover gradients over short
  trajectories, not whole training runs. The confidence intervals in the tables show the spread
  across training seeds, the spread to expect between a rerun and the committed values.

## Where every output is written

| Output | Directory | Written by |
|---|---|---|
| G2 regression: per-seed results, `search.json`, `search/`, `table.md`, plots | `results/regression/g2/full/` | `ebt.experiments.regression_g2`, `tools.regression_g2_report` |
| MNIST denoising: per-seed results, `search.json`, `search/`, `table.md`, plots | `results/mnist/denoising/full/` | `ebt.experiments.mnist_denoising`, `tools.mnist_report` |
| Text: per-seed results, `search.json`, `search/`, `table.md`, `samples.md`, plot | `results/text/lm/full/` | `ebt.experiments.text_lm`, `tools.text_report` |
| Text budget and timing measurements | `results/text/measurements-2026-11-20/`, `results/text/measurements-2026-11-21/` | see [Every published file](#every-published-file-and-how-it-is-made) |
| G3 and G4: per-seed results, `summary.json`, `table.md` | `results/<task>/g3_g4/full/` | `ebt.experiments.inference_g3_g4` |
| G5: per-seed results, `summary.json`, `g5.json`, `table.md` | `results/<task>/g5/full/` (MNIST: `noise-<level>/` per level) | `ebt.experiments.inference_g5`, `tools.g5_report` |
| G3 figure | `results/g3_steps.png` | `tools.g3_figure` |
| Checkpoints | `checkpoints/<experiment>/full/` (not committed) | the training experiments |

## Figures

The PNG files under `results/`, each generated by the report tool of its directory, except
`results/g3_steps.png`, which `python -m tools.g3_figure` draws from the three G3 summaries:

| File | What it shows | Described in |
|---|---|---|
| `results/regression/g2/full/predictions.png` | the test data and the predictions of every model against x, seed 0 | [method.md 16.7](method.md#167-metrics) |
| `results/regression/g2/full/energy_ebt_mse.png`, `energy_ebt_contrastive.png`, `energy_ebt_combined.png`, `energy_ebt_regularized.png` | E(x, ŷ) minus its minimum over ŷ at each x for (b), (c1), (c2) and (c3), seed 0 | [method.md 16.7](method.md#167-metrics) |
| `results/regression/g2/full/energy_ibc.png` | the same for (c4) | [method.md 16.8](method.md#168-models-from-outside-the-ebt-paper-a2-c4-and-c5) |
| `results/regression/g2/full/energy_ired.png`, `energy_ired_k9.png` | the same for (c5), on landscape k = 0 and on the smoothest landscape k = 9 | [method.md 16.8](method.md#168-models-from-outside-the-ebt-paper-a2-c4-and-c5) |
| `results/regression/g2/full/density_mdn.png` | −ln p(ŷ \| x) minus its minimum over ŷ for (a2), seed 0 | [method.md 16.8](method.md#168-models-from-outside-the-ebt-paper-a2-c4-and-c5) |
| `results/mnist/denoising/full/examples.png` | test images of seed 0: clean, noisy input, the one-pass denoiser's output and the EBT's output after each step | [method.md 17.6](method.md#176-metrics) |
| `results/mnist/denoising/full/validation.png` | validation PSNR and Smooth L1 against the training step, one line per seed | [method.md 17.6](method.md#176-metrics) |
| `results/text/lm/full/validation.png` | validation bits per character against the training step, one line per seed | [method.md 18.9](method.md#189-results-of-the-final-runs) |
| `results/g3_steps.png` | G3: text BPC, MNIST PSNR and regression r of (b), (c1), (c2) and (c3) against 1 to 64 inference steps, mean over seeds 0–4 with the 95% t-interval as a band and the trained step count marked; regression r on a log scale | [method.md 19.5](method.md#195-results) |

## Durations and hardware

v1.0 ran on one Apple M5 Mac with 16 GB of unified memory: regression on the CPU, MNIST and text
on MPS. Recorded durations (field `seconds` of each per-seed result file):

- G3/G4 evaluation (`results/<task>/g3_g4/full/`): text 457 to 458 s per seed (2 287 s for 5
  seeds); MNIST 275 to 277 s per seed (1 382 s); regression under 20 s per seed for every model
  except IBC (c4), 560 to 599 s per seed (2 943 s for 5 seeds).
- G5 evaluation (`results/<task>/g5/full/`): text 185 to 215 s per seed (1 009 s); MNIST 151 to
  153 s per seed and level (2 283 s for 15 runs); regression under 1 s per checkpoint.
- Text training: the result files record no duration. The projection of method.md section 18.8,
  from measured step times, is 3.19 h for the search and 4.03 h for the final runs
  (`results/text/measurements-2026-11-21/budget/choice.json`, `search_hours` and `final_hours`).
- Regression and MNIST training: their durations were not recorded.

## Tests

```sh
ruff check .
ruff format --check .
pytest                 # every test; the MPS tests are skipped where MPS is unavailable
pytest -m "not mps"    # the CPU suite: the MPS tests deselected
```

On the Apple M5 machine of v1.0 the CPU suite (`pytest -m "not mps"`) has 467 tests, which passed
in 403 s on 24 November 2026, with 10 deselected. The deselected tests are the `mps` tests, in
more than one file (`tests/test_mps.py`, `tests/test_text_training.py`). On a machine without MPS
a plain `pytest` skips them.
