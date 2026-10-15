# CSTI-ReID — **Paper:** *Causal-Inspired Style Intervention for Source-Free Test-Time Adaptation in Domain-Generalizable Person Re-Identification*.



|Paper component|Location|Behavior|
|-|-|-|
|ViT-B/16 encoder + domain tokens|`models/encoder.py`|Non-square 256×128 crops, learnable tokens, AdaIN at the first three transformer blocks, LayerNorm-only online updates|
|Source pretraining|`train.py`, `models/source\_trainer.py`|Source-only identity cross-entropy and soft-margin hard triplet, AdamW cosine learning rate|
|Online occupancy style bank|`models/csti\_reid.py`|Fixed capacity, cosine novelty, preserved anchor, active counts, oldest-slot eviction|
|IWBRE|`models/csti\_reid.py`|Query-biased style proposal, AdaIN branches, with-replacement sampling, self-normalized importance correction|
|RNCC|`models/csti\_reid.py`|Branch-pair cosine agreement|
|MUC|`models/csti\_reid.py`|Normalized prototype entropy × intervention-sensitivity reliability. Fixed pre-batch memory snapshot, batch normalization, absolute B=1 fallback|
|DPPM|`models/csti\_reid.py`|Post-SGD sequential admissions, novelty, reliability and consistency gates, prototype and utility EMAs, oldest tie-break|
|Style-occupancy, entropy, alignment losses|`models/csti\_reid.py`|Differentiable current-style KL and detached prototype targets|
|Fixed-gallery query-only evaluation|`evaluate.py`|Source gallery cached once; query ranks computed before gradient update|
|Protocols A, B, C|`evaluate.py`, `protocols/`|Separate targets, nested Protocol-B source-only fold plan/selection, camera-specific Protocol-C checkpoints and macro-average|
|Scientific statistics|`utils/stats.py`, `protocols/aggregate.py`|Sample SD (ddof=1), paired results only with matching seeds, no fabricated p-values|
|Separate identity-safe theory|`models/robust.py`|Identity-anchor selection, KL robust finite-slot dual, drift acceptance utility. Not included in base headline results|
|Ablations|`evaluate.py --variant`|`full`, `simple\_sa`, `no\_iwbre`, `no\_rncc`, `no\_muc`, `no\_dppm`|
|Unit/integration checks|`tests/`|Mathematical invariants, source/target separation, online update order, corner cases|

## Install

Python 3.10+ recommended; PyTorch CUDA matching your GPU environment.

```bash
python -m pip install -r requirements.txt
# optional cross-backbone DeiT-S
python -m pip install timm
python -m pytest -q
```

## Input format

Use per-dataset **CSV manifests** with `path,pid,cam,split` columns, where `split` is `train`, `query`, or `gallery`. Image paths can be relative to the CSV directory. **Official identity-disjoint splits are required**. Market-1501/Duke folder parsing is also supported in `data/datasets.py`. Prepare `CUHK03`, `MSMT17`, and small-dataset manifests from your legitimately obtained release and recorded test protocols. No benchmark images are bundled.

Run preflight:

```bash
python -m protocols.audit\_manifests --name market1501 \\
  --manifest /data/market1501.csv --output outputs/market\_audit.json
```

## Source-only training — Protocol A

The paper's default ViT-B/16 config is `configs/paper.yaml`. For the **exact reported paper results**, use the same saved source weights and ImageNet-21k pretraining source. The default `pretrained: false` prevents accidentally substituting torchvision's ImageNet-1k weights for the reported initialization.

```bash
python train.py --config configs/paper.yaml \\
  --source-name market1501 --source-manifest /data/market1501.csv \\
  --save-dir checkpoints/protoA\_market --device cuda
```

## Protocol A — online target stream

```bash
python evaluate.py --help
# Real example (replace dataset paths):
python evaluate.py --protocol A --target msmt17 \\
  --target-manifest /data/msmt17.csv \\
  --checkpoint checkpoints/protoA\_market/source\_best.pt \\
  --config configs/paper.yaml --n-runs 5 \\
  --output outputs/protocol\_A\_msmt17.json
```

## Protocol B — nested source-only model selection

```bash
python -m protocols.fold\_plan --output outputs/protocol\_b\_fold\_plan.json
```

Each outer target must be excluded from source training, source style pools, checkpoint selection, and hyperparameter tuning. Train separate multi-source checkpoints with `--outer-target` and only the **three remaining** `--source-name` / `--source-manifest` pairs. For inner-fold tuning, train with two eligible sources and validate on the third, using only its permitted train-identity development split. Once all three inner-fold validation results have been recorded in the documented JSON structure:

```bash
python -m protocols.nested\_select --inner-results outputs/inner\_results.json \\
  --output outputs/selected\_source\_only\_config.json
python evaluate.py --protocol B --target market1501 \\
  --target-manifest /data/market1501.csv \\
  --checkpoint checkpoints/protoB\_market/source\_best.pt \\
  --output outputs/protocol\_B\_market.json
```

The evaluation tool rejects unaudited or target-contaminated Protocol-B checkpoint metadata. It does **not** perform the expensive inner-fold training automatically.

## Protocol C — camera and identity separation

For a target camera `c`, train a **different source encoder** with `--exclude-camera c`, using only official *training identities* from the remaining cameras. The gallery is used exclusively to produce the fixed retrieval cache. Query labels/cameras are available only to the post-hoc evaluator. Camera folds start from new model, optimizer, style bank and prototype memory. Results are the unweighted macro-average over eligible camera folds.

```bash
python train.py --config configs/paper.yaml \\
  --source-name market1501 --source-manifest /data/market1501.csv \\
  --exclude-camera 1 --save-dir checkpoints/protoC\_market/camera\_1
# Repeat once per eligible target camera.
python evaluate.py --protocol C --target market1501 \\
  --target-manifest /data/market1501.csv \\
  --checkpoint-pattern 'checkpoints/protoC\_market/camera\_{camera}/source\_best.pt' \\
  --output outputs/protocol\_C\_market.json
```

A camera's official gallery images can remain available for **retrieval only**. Images with the same identity and same camera as the query are filtered from its ranking. Queries without cross-camera positives are excluded from mAP/Rank-1. No held-out camera is used in labeled source development.

