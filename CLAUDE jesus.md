# CLAUDE.md — ProjectDSAI2026_2 (jesussebastian)

## Project context

Telecom Paris 2A SD project. Benchmark suite for YOLO-family object-detection
models, run in parallel with other student teams (e.g. Antoine → DETR family)
for cross-model comparison. Results must follow a shared protocol so numbers
are directly comparable across teams.

Owner tag: **jesussebastian**
Branch: **jesus_sebastian**
Repo: `gitlab.enst.fr:dsai/projects-2026/ProjectDSAI2026_2`
Runtime: Google Colab T4 GPU (16 GB GDDR6, FP32 eager mode)

> **Folder rename** (commit `da19cda`): all three top-level output namespaces
> were renamed — `optim/yolo/` → `optim/yolox_v7_damo/`, `results/yolo/` →
> `results/yolox_v7_damo/`, `outputs/jesussebastian/` → `outputs/yolox_v7_damo/`.
> The Python scripts inside `optim/yolox_v7_damo/` still contain the OLD path
> literals (`results/yolo/optimized`, `OWNER = "jesussebastian"`) because they
> were written before the rename — they must be updated before a fresh Colab run.

---

## Models benchmarked

| Name | Architecture | Checkpoint source |
|---|---|---|
| `yolox_m` | YOLOX-m (YOLOPAFPN + YOLOXHead) | Megvii-BaseDetection/YOLOX 0.1.1rc0 |
| `yolox_l` | YOLOX-l | same |
| `yolox_x` | YOLOX-x | same |
| `damo_yolo_m` | DAMO-YOLO-M (TinyNASL35 + GiraffeNeck + ZeroHead) | HuggingFace Cwhgn/DAMO-YOLO-M |
| `yolov7` | YOLOv7 (flat nn.Sequential, RepConv fused) | WongKinYiu/yolov7 v0.1 |
| `yolov7x` | YOLOv7-X | same |

---

## Benchmark protocol (exact, cross-team comparable)

- **GPU**: NVIDIA Tesla T4, FP32, eager mode (no torch.compile / TensorRT / quantization)
- **Input**: single 640×640 image, index-0 of the seed=42 2000-image COCO val2017 subset,
  preprocessed per-model (YOLOX → BGR [0,255]; YOLOv7 → RGB [0,1]; DAMO → RGB [0,255])
- **Latency**: 50 warmup + 1000 timed `model(x)` calls, each bracketed by
  `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()`
- **Seeds**: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`,
  `torch.cuda.manual_seed_all(42)` at the top of every `main()`
- **Post-processing**: included in forward pass (YOLO decode is inside `model.eval()`)
- **mAP**: 2000-image seed=42 COCO val2017 subset, via `utils.coco_eval.evaluate_map`

---

## Key files

### Benchmark scripts (committed, protocol-correct)
```
models/yolo/<model>/bench.py     # 6 scripts — latency + mAP, one per model
models/yolo/run_all_baselines.py # runs all 6 sequentially, writes baseline_yolo_family.csv
models/yolo/_yolo_coco_dataset.py # CocoYoloDataset, get_coco_val_subset_yolo(), letterbox
```

### Model loaders
```
models/yolo/<model>/load.py      # load_model() — downloads checkpoint on first run
models/yolo/yolov7/load.py       # patches torch.hub + calls model.fuse() for RepConv
models/yolo/damo_yolo_m/load.py  # clones DAMO-YOLO repo to vendor/ on first run
```

### Utilities
```
utils/loop_bench.py              # benchmark_inference() — the timed harness
utils/coco_loader.py             # CocoSubsetDataset, get_fixed_image() (DETR-style)
utils/coco_eval.py               # evaluate_map()
```

### Optimization scripts (S3 phase — inference-only, all committed)
```
optim/yolox_v7_damo/shared/conv_bn_fusion.py     # fuse_conv_bn() via torch fuse_conv_bn_eval
optim/yolox_v7_damo/shared/amp_wrapper.py        # AMPWrapper — autocast(fp16), FP32 in/out
optim/yolox_v7_damo/shared/cuda_graph_wrapper.py # CUDAGraphWrapper — manual capture/replay (yolox_m only)
optim/yolox_v7_damo/shared/compile_util.py       # attempt_compile() — torch.compile ro→def→eager, reports compile_mode
optim/yolox_v7_damo/shared/tensorrt_wrapper.py   # TensorRTWrapper — ONNX→TRT FP16 engine, FP32 in/out
optim/yolox_v7_damo/shared/stage_profiler.py     # per-stage torch.profiler pass + TRT IProfiler path;
                                                  #   profile_stage_ops()       → stage_ops/<stage>_ops.csv
                                                  #   profile_trt_stage_ops()   → real per-layer TRT CSV via trt.IProfiler
                                                  #   compute_speedup_factors() → op_speedup_factors.csv + _meta.csv
                                                  #   build_sequential_latency_table() → op_latency_sequential.csv
                                                  #   build_cross_model_sequential_table() → _ALL.csv
                                                  #   leaf_op_label()           → protocol op taxonomy (same as notebook)
optim/yolox_v7_damo/shared/stage_names.py        # MODEL_STAGE_NAMES dict + CANONICAL_STAGES list (torch-free;
                                                  #   mirrors the STAGE_NAMES literal in each optimize.py)
optim/yolox_v7_damo/<model>/optimize.py          # 6 per-model pipelines (stages → JSON)
optim/yolox_v7_damo/run_all_optimizations.py     # runs all 6, writes optimization_summary.csv + cross_model_op_speedup.csv
optim/yolox_v7_damo/recompute_op_tables.py       # CPU-only: rebuilds op_speedup_factors / op_latency_sequential CSVs
                                                  #   from existing stage_ops CSVs without re-profiling (no GPU/torch)
optim/yolox_v7_damo/damo_yolo_m/trt_deploy_export.py  # deploy-mode export: bypasses DAMO head NMS for cleaner ONNX graph
optim/yolox_v7_damo/damo_yolo_m/trt_deploy_smoke.py   # smoke test for the deploy-mode engine
optim/yolox_v7_damo/tests/test_op_tables.py      # asserts stage_names.py MODEL_STAGE_NAMES matches each optimize.py literal
optim/yolox_v7_damo/requirements-optim-yolo.txt  # general optim deps
optim/yolox_v7_damo/requirements-tensorrt.txt    # torch-tensorrt 2.4.0 + onnx (TRT stage only)
```
`tensorrt_wrapper.py`, `compile_util.py`, and `stage_profiler.py` are imported directly
(e.g. `from optim.yolox_v7_damo.shared.stage_profiler import profile_stage_ops,
compute_speedup_factors`) — `shared/__init__.py` is part of the frozen contract and is
NOT modified. `stage_names.py` is also loaded directly (via importlib in
`recompute_op_tables.py`) to avoid triggering the torch-dependent `__init__.py` on
CPU-only machines.

### Main notebook
```
utils/notebooks/05_run_yolo_benches.ipynb
```

Key cells in the notebook:
- **pip cells** (`a6c3eff5`, `c63b1148`): install yolox, loguru, ninja, thop
- **cell-2**: set PROJECT_ROOT, verify utils exist
- **Option B** (`cell-7`): `!python -m models.yolo.run_all_baselines` — ~30 min, produces mAP + latency JSON
- **yolo-breakdown-code**: coarse backbone/neck/head CUDA-event breakdown (display only)
- **pip cell** (`tensorrt-pip`): OPTIONAL TensorRT install for the TRT stage — installs the **standalone `tensorrt`** wheel (+ `onnx`), NOT `torch-tensorrt`, so it does **not** touch torch/torchvision and needs **no runtime restart**. Lives at the **END** of notebook 05 (NOT in the setup region). Skip for baseline/protocol.
- **nsys cell** (`aec03edd`): Nsight Systems kernel profiling — intermittent on Colab, run before any torch.profiler cell. Also copies each `{model}_nsys_kernel_breakdown.png` into `outputs/yolox_v7_damo/<model>/charts/` and the global `backbone_device_attribution_nsys.png` into every model's `charts/` (so the zip picks them up); the originals still go to `results/yolox_v7_damo/`.
- **Protocol cell** (`063efe07`): THE main deliverable cell — runs per model:
  latency timing + torch.profiler breakdown + fine-grained charts + all CSVs
- **Zip cell** (`9e5b0374`): zips `outputs/yolox_v7_damo/` and downloads
- **Drive backup cell** (`184258e2`): copies zip to Google Drive

---

## Deliverable tree (per model)

```
outputs/yolox_v7_damo/<model>/
├── config.yaml
├── results/
│   ├── latencies.csv               # raw 1000-iter timings
│   ├── summary.csv                 # mean/std/p50/p95/p99/fps/params
│   ├── submodules.csv              # backbone/neck/head % breakdown
│   ├── backbone_ops.csv            # Conv1x1/Conv3x3/BatchNorm2d/SiLU/... per-op ms
│   ├── system_breakdown.csv        # gpu_compute / mem_copies / cpu_launch / wall
│   ├── kernels.csv                 # top-5 CUDA kernels (nsys) or aten ops (fallback)
│   ├── op_latency_sequential.csv   # Phase 2: per-op ms + vs_baseline + vs_previous_stage factors for every stage
│   ├── op_speedup_factors.csv      # per-op speedup table (baseline/stage >1=faster), Conv_total headline row, explicit sentinels
│   └── op_speedup_factors_meta.csv # per-stage profiling granularity (per_op / fused_graph_total_only / trt_coarse_buckets / ...)
├── stage_ops/
│   ├── baseline_ops.csv            # per-op CUDA self-times for each stage
│   ├── amp_fp16_ops.csv
│   ├── torch_compile_ops.csv
│   ├── tensorrt_fp16_ops.csv
│   └── ...                         # one CSV per stage; yolox_m also has conv_bn_fusion_ops.csv, cuda_graphs_ops.csv
├── traces/
│   └── <model>_trace.json          # chrome://tracing compatible
├── nvidia_profiles/
│   ├── nsys_benchmark.nsys-rep
│   └── nsys_stats.txt
└── charts/
    ├── stacked_latency.png                  # backbone/neck/head stacked bar
    ├── backbone_ops.png                     # fine-grained: Conv1x1, Conv3x3, BatchNorm2d, SiLU, etc.
    ├── <model>_nsys_kernel_breakdown.png    # nsys per-op GPU time (copied from results/yolox_v7_damo/ by the nsys cell)
    └── backbone_device_attribution_nsys.png # nsys GPU compute vs memcpy — GLOBAL chart, same file copied into every model's charts/
```

---

## How to run in a new Colab session (baseline + protocol)

1. Run `00_setup_colab.ipynb` — mounts Drive, clones repo, installs deps (~3 min)
2. In `05_run_yolo_benches.ipynb`: run pip cells (`a6c3eff5`, `c63b1148`) + cell-2
3. **Optional:** Run **Option B** (`cell-7`) — `!python -m models.yolo.run_all_baselines` (~30 min, produces FP32 baseline mAP)
4. Run **Protocol cell** (`063efe07`) — ~10–15 min, generates all deliverables (latency, charts, torch.profiler traces)
5. Run **Zip cell** (`9e5b0374`) or **Drive backup cell** (`184258e2`) to download

To also include TensorRT results: run steps 6–8 below (separate pipeline after protocol is done).

---

## Optimization phase (S3) — `optim/yolox_v7_damo/`

Inference-only optimizations layered on top of the FP32 baselines. Each model has
its own `optimize.py` that benchmarks the baseline, then each optimization stage,
writes `results/yolox_v7_damo/optimized/<model>.json` (one entry per stage), and measures
mAP once on the fully optimized model. `run_all_optimizations.py` aggregates all
six into `results/yolox_v7_damo/optimized/optimization_summary.csv` (one row per (model, stage),
columns: model, stage, opt_applied, elementary_op_changed, mean_ms, fps, mAP,
speedup_vs_baseline, status). It also writes `cross_model_op_speedup.csv` and
`op_latency_sequential_ALL.csv`.

The **printed stdout** from `run_all_optimizations.py` shows a `native_fin` column
(best PyTorch-native stage, normally torch_compile). That label does **not** appear
as a CSV column — read it from the last non-`tensorrt_fp16` row for each model in
`optimization_summary.csv`.

**Frozen contract**: optimize.py never modifies `load.py` / `bench.py` /
`_yolo_coco_dataset.py` / `utils/*`. Post-processing adapters are *imported* from
each `bench.py` (`yolox_adapter`, `yolov7_adapter`, `damo_yolo_adapter`). The AMP
and CUDA-Graph wrappers are `nn.Module` subclasses because `benchmark_inference`
and `evaluate_map` both call `next(model.parameters())` + `model.eval()` (a bare
callable would fail). Baseline mean-latency constants for the speedup column are
hardcoded in `run_all_optimizations.py` (baseline result JSONs are gitignored).

### Per-model optimization map

| Model | Conv+BN fusion | AMP FP16 | CUDA Graphs | torch.compile | TensorRT FP16 |
|---|:--:|:--:|:--:|:--:|:--:|
| yolox_m | ✅ | ✅ | ✅ manual (attempted) | ✅ ro→def³ | ✅ (mAP measured) |
| yolox_l | ✅ | ✅ | via compile³ | ✅ ro→def³ | ✅ (mAP measured) |
| yolox_x | ✅ | ✅ | via compile³ | ✅ ro→def³ | ✅ (mAP measured) |
| yolov7 | already at load¹ | ✅ | via compile³ | ✅ ro→def³ | ✅ (mAP skipped²) |
| yolov7x | already at load¹ | ✅ | via compile³ | ✅ ro→def³ | ✅ (mAP skipped²) |
| damo_yolo_m | ✅ (stage 3) | ✅ (stage 4) | via compile³ | ✅ (stage 1) ro→def³ | ✅ (mAP measured) |

**Every model now carries a torch.compile stage (universal, for cross-model comparability).**

¹ yolov7/yolov7x: `load.py` already calls `model.fuse()` (RepConv reparam + Conv/BN
fold) — no BN left, never re-fuse.

² **TensorRT** is the NEW final stage. It is always built from a **fresh
`load_model()`** (pristine FP32) — never the in-place-fused / AMP / compiled object,
because TRT does its own Conv+BN fold + FP16 selection and needs to see the canonical
`Conv→BN→activation` pattern. mAP via the imported adapter only works for **YOLOX** and
**DAMO** (single-tensor or reconstructible output); **YOLOv7** (tuple) outputs cannot be
reconstructed from a raw engine, so those record `mAP: null` +
`mAP_status="skipped_output_format_mismatch"`. A genuine eval error →
`failed_runtime_error`. Latency timing is unaffected (never inspects the output).

³ **torch.compile** is now a universal stage on every model via the shared
`attempt_compile` (`optim/yolox_v7_damo/shared/compile_util.py`, mirrors Ultralytics'
`attempt_compile`): `torch.compile(mode="reduce-overhead", backend="inductor")`
first — Inductor op-fusion **+ CUDA Graphs** (the launch-overhead collapse, the
only win for flat YOLOv7/x at batch=1) — falling back to `mode="default"`
(Inductor fusion only) then eager. The mode that actually landed is recorded as
`compile_mode` in each model's JSON, so the write-up can tell the launch-overhead
path (reduce-overhead) from the op-fusion path (default) from a no-op (eager). On
yolox_m this is the AUTOMATIC counterpart to the MANUAL `CUDAGraphWrapper` stage;
because `compile_mode` is recorded, the two never double-claim the same win (manual
capture fails on dynamic shapes → reduce-overhead's internal capture hits the same
wall → `compile_mode` drops to `default`). "via compile³" in the table = CUDA Graphs
are reached through reduce-overhead, not a separate manual stage.

### Measured results (T4, batch=1)

> Numbers from the latest `run_all_optimizations` Colab run. Baseline ms is the
> **re-measured** FP32 baseline inside the optimization run (slightly differs from
> the original bench run due to Colab session state). `speedup_vs_baseline` in the
> CSV is computed against hardcoded BASELINE_MEAN_MS constants (21.8 / 41.8 / 75.6 /
> 33.9 / 60.7 / 25.3 ms) — the values in `run_all_optimizations.py`'s `BASELINE_MEAN_MS`
> dict — NOT the re-measured baseline below. mAP is from the last stage that measured it.

| Model | Baseline ms | AMP ms | AMP sp. | Compile ms | Compile sp. | TRT FP16 ms | TRT sp. | mAP |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| yolox_m | 22.27 | 17.93 | 1.22× | 8.61 (reduce-overhead) | 2.53× | 6.02 | 3.62× | 0.4745 |
| yolox_l | 47.67 | 22.43 | 1.86× | 13.69 (reduce-overhead) | 3.05× | 10.17 | 4.11× | 0.5051 |
| yolox_x | 83.09 | 34.32 | 2.20× | 24.38 (reduce-overhead) | 3.10× | 18.17 | 4.16× | 0.5229 |
| yolov7 | 37.99 | 17.47 | 1.94× | 18.16⁵ (failed→AMP) | 1.87×⁵ | 8.34 | 4.07× | 0.5190 |
| yolov7x | 65.77 | 27.37 | 2.22× | 27.97⁵ (failed→AMP) | 2.17×⁵ | 14.08 | 4.31× | 0.5374 |
| damo_yolo_m | 26.52 | 9.90 | 2.56× | 25.96⁶ (compile≈baseline) | 0.97×⁶ | 6.39 | 3.96× | 0.5066 |

⁵ yolov7/v7x: torch.compile failed in all modes (flat `nn.Sequential` gives inductor
nothing to trace); eager model kept — compile ms ≈ AMP ms (same model, measurement noise).

⁶ damo_yolo_m: torch.compile landed `reduce-overhead` (0.974×, essentially a no-op at
batch=1); AMP at 9.90ms is the effective native-final. The TRT stage uses a fresh
`load_model()` and achieves 3.96× over the hardcoded baseline.

No mAP degradation on any model. DAMO TRT export succeeded (standard ONNX path, contrary
to early expectations). Conv+BN fusion alone is a wash on T4 (see caveats).

### TensorRT FP16 — measured results

| Model | Baseline ms¹ | native-final ms | TRT FP16 ms | TRT speedup¹ | mAP_status |
|---|--:|--:|--:|--:|---|
| yolox_m | 21.8 | 8.61 (torch_compile) | 6.02 | 3.619× | ok (mAP=0.4745) |
| yolox_l | 41.8 | 13.69 (torch_compile) | 10.17 | 4.111× | ok (mAP=0.5051) |
| yolox_x | 75.6 | 24.38 (torch_compile) | 18.17 | 4.16× | ok (mAP=0.5229) |
| yolov7 | 33.9 | 18.16 (AMP, compile failed) | 8.34 | 4.065× | skipped_output_format_mismatch |
| yolov7x | 60.7 | 27.97 (AMP, compile failed) | 14.08 | 4.311× | skipped_output_format_mismatch |
| damo_yolo_m | 25.3 | 9.90 (AMP, compile≈no-op) | 6.39 | 3.961× | ok (mAP=0.5066) |

¹ speedup_vs_baseline in `optimization_summary.csv` is computed against the hardcoded
`BASELINE_MEAN_MS` constants in `run_all_optimizations.py`, NOT the re-measured baseline.

**damo_yolo_m TRT also has a deploy-mode variant**: `trt_deploy_result.json` records
`status=ok`, `export_mode=deploy_head` (NMS/decode bypassed), mean_ms=6.04,
speedup_vs_baseline=4.391×; mAP is null (raw tensor output incompatible with adapter).
The standard pipeline row (6.39ms) successfully exported with NMS in the graph and
measured mAP; the deploy-mode is a separate artifact, not in `optimization_summary.csv`.

### Cross-model per-op speedup (averaged across models)

From `results/yolox_v7_damo/optimized/cross_model_op_speedup.csv`
(TRT factors are REAL per-layer measurements via `trt.IProfiler`; `trt_profiler="IProfiler"`):

| op_type | avg AMP | avg compile | avg TRT | n_amp | n_compile | n_trt |
|---|--:|--:|--:|--:|--:|--:|
| Conv_total | 1.645 | 1.997 | 3.056 | 5 | 2 | 6 |
| Conv1x1 | 1.312 | 1.839 | 2.156 | 5 | 2 | 1 |
| Conv3x3 | 1.913 | 2.086 | — (TRT fused) | 5 | 2 | 0 |
| MaxPool2d | 1.368 | 1.535 | 2.647 | 5 | 2 | 4 |
| SiLU | 2.224 | 2.090 | 4.529 | 5 | 2 | 2 |
| Upsample | 0.745 | — | 0.917 | 1 | 0 | 1 |

`avg compile` is low n_models (2) because torch.compile usually collapses to
`fused_or_replayed_graph` (no per-op hooks fire); only the 2 models whose compile
landed on `mode=default` (not reduce-overhead) produce per-op rows.
`Conv_total` = sum of all conv-like ops — the one row comparable across all stages
including TRT (which fuses Conv3x3 away into the engine).

### How to run (Colab, separate TensorRT optimization phase)

After the **Protocol cell** finishes (steps 1–4 above), run these steps:

6. Run the **tensorrt-pip cell** (at the end of notebook 05) — installs the standalone `tensorrt` + `onnx`.
   ✅ **No runtime restart needed**: the standalone `tensorrt` wheel has no torch dependency (torch/torchvision stay pinned, benches keep working), and the optimization cell runs in a fresh `python` subprocess that picks up the new package automatically.
7. Run the **optimization cell** (new cell you add):
   ```python
   %cd /content/project-dsai
   !python -m optim.yolox_v7_damo.run_all_optimizations      # all 6 → optimization_summary.csv + summary table
   ```
   This produces:
   - `results/yolox_v7_damo/optimized/<model>.json` (per-model stage details + TRT numbers)
   - `results/yolox_v7_damo/optimized/optimization_summary.csv` (one row per (model, stage))
   - `results/yolox_v7_damo/optimized/cross_model_op_speedup.csv`
   - `results/yolox_v7_damo/optimized/op_latency_sequential_ALL.csv`
   - Per-model: `results/yolox_v7_damo/optimized/<model>/stage_ops/<stage>_ops.csv`
   - Per-model: `outputs/yolox_v7_damo/<model>/op_latency_sequential.csv` + `op_speedup_factors.csv` + `_meta.csv`
   - Prints the baseline→native-final→TRT speedup table (with `native_fin` column) to stdout
8. (Optional) Copy optimization results into the deliverable tree so the final zip includes them:
   ```python
   import shutil
   from pathlib import Path
   src = Path("/content/project-dsai/results/yolox_v7_damo/optimized")
   dst = Path("/content/project-dsai/outputs/yolox_v7_damo/_optimization_results")
   # Copy ONLY the human-readable results (optimization_summary.csv, per-model JSON,
   # logs). The TensorRT engines (*.trt/*.ep/*.engine) and ONNX export intermediates
   # (*.onnx + *.onnx.data sidecar) are huge binary build artifacts — leaving them in
   # bloats the deliverable zip to ~1 GB, so they are ignored here.
   shutil.copytree(src, dst, dirs_exist_ok=True,
                   ignore=shutil.ignore_patterns("*.onnx", "*.onnx.data", "*.trt",
                                                  "*.trt.ep", "*.ep", "*.engine", "*.plan"))
   print(f"Copied to {dst}")
   ```
9. Run the **Zip cell** (`9e5b0374`) to download (now includes optimization results + protocol deliverables)
   The zip cell **also** skips `*.onnx`/`*.trt`/`*.ep`/`*.engine` build artifacts as a
   safety net (and prints how many it skipped), so the bundle stays small even if step 8
   was run with the old un-filtered copy.

**To rebuild per-op tables without re-running on GPU** (e.g. after a `stage_profiler.py`
logic fix): `python -m optim.yolox_v7_damo.recompute_op_tables` — reads existing
`stage_ops/<stage>_ops.csv` files and regenerates `op_speedup_factors.csv`,
`op_speedup_factors_meta.csv`, and `op_latency_sequential.csv` per model, then stacks all
six into `op_latency_sequential_ALL.csv`. Cells for missing stage CSVs come out as the
explicit `stage_not_run` sentinel, never blank.

**TRT timing**: the `tensorrt_fp16` stage adds **~3–5 min per model on the first run**
(ONNX export + engine auto-tuning) and **~10 s on a cached run** — the serialized
engine is written to `results/yolox_v7_damo/optimized/<model>/engine_fp16.trt` and reloaded if
it is newer than that model's `load.py`. A changed `load.py` (weights/preprocessing)
invalidates the cache and forces a rebuild. Engine files are gitignored.

**Why a separate phase?** The Protocol cell must stay pure FP32 (the official baseline
for cross-team comparison). TensorRT optimizations are measured *on top of* the
baseline, not instead of it. The optimization phase can also change torch/torchvision
ABI, so it runs in a separate runtime session.

---

## Per-model preprocessing (for `get_coco_val_subset_yolo`)

| Model | `channel_order` | `scale` |
|---|---|---|
| yolox_m/l/x | `"bgr"` | `1.0` (stays [0,255]) |
| damo_yolo_m | `"rgb"` | `1.0` (stays [0,255]) |
| yolov7, yolov7x | `"rgb"` | `1/255` (→ [0,1]) |

---

## Known issues / caveats

- **nsys CUPTI conflicts**: nsys profiling fails intermittently on Colab when
  `torch.profiler` ran in the same kernel first. Always run the nsys cell
  **before** the protocol cell on a fresh kernel. Retry cell provided in session.
- **yolov7/yolov7x backbone chart (notebook)**: these models are flat `nn.Sequential` with
  no `.backbone` attribute — `backbone_ops.png` from the Protocol cell shows "No backbone
  leaves tagged", which is expected and correct. In the **S3 stage profiler**
  (`stage_profiler.py`), this is handled with `use_global_fallback=True`: every leaf of the
  whole model is tagged, so `stage_ops/baseline_ops.csv` etc. DO exist and contain full
  per-op breakdowns for these models.
- **DAMO-YOLO vendor clone**: `load_model()` clones the DAMO-YOLO repo to
  `models/yolo/damo_yolo_m/vendor/` (gitignored) on first run. Needs internet.
- **Colab disk is ephemeral**: all files in `/content/` are wiped on runtime
  disconnect. Only Google Drive persists. Always save to Drive after a run.
- **COCO dataset location**: `/content/drive/MyDrive/dsai_coco/` on Drive.
  `val2017/` images + `annotations/instances_val2017.json` must be present.
- **YOLOv7 requirements warning**: `torch.hub.load` prints numpy/protobuf
  version warnings — harmless, model loads and benchmarks correctly.
- **S3 / Conv+BN fusion is a wash on T4**: folding is mathematically correct but
  gives no measurable speedup (0.95–0.97×, within noise) — cuDNN's batch-norm is
  already well-optimized and the block is memory-bandwidth bound, not launch bound.
  Kept in the pipeline for completeness / the elementary-brick write-up.
- **S3 / Conv+BN accounting artifact in per-op tables**: when BatchNorm is folded into
  Conv at the `conv_bn_fusion` stage, BN's CUDA self-time relocates into the Conv bucket,
  making Conv appear slower at that stage (and faster in BN). The `Conv_total` row
  (sum of all conv-like ops) is the trustworthy signal; individual `Conv3x3`/`Conv1x1`
  factors AT the fusion stage should not be interpreted as a regression — they reflect
  the BN time now counted inside Conv. `op_speedup_factors.csv` marks the BN cells with
  the explicit `fused_into_conv` sentinel from `conv_bn_fusion` onward.
- **S3 / op_latency_sequential.csv sentinels**: cells that cannot be measured are
  marked explicitly — `fused_into_conv` (BatchNorm folded), `opaque_compiled_graph`
  (compiled/CUDA-graph region not decomposable by torch.profiler forward hooks),
  `trt_fused_not_separable` (TensorRT merged op into engine; use Conv_total instead),
  `below_threshold` (profiled but under 0.1 ms noise floor), `stage_not_run` (CSV
  absent), `stage_not_in_pipeline` (cross-model only: model never runs this stage).
  No silent blank cells.
- **S3 / CUDA Graphs capture fails on yolox_m**: the forward has dynamic
  shapes/control flow capture can't handle. `CUDAGraphWrapper` raises a clear error;
  `optimize.py` catches it and keeps the AMP model as final (`cuda_graph_fallback_reason`).
- **S3 / torch.compile fails on yolov7 + yolov7x**: the flat `nn.Sequential` deploy
  graph gives TorchInductor nothing to fuse/break on; caught + logged
  (`compile_fallback_reason`), AMP model kept as final. It *succeeds* on
  damo_yolo_m but gives only ~0.97× (essentially no-op at batch=1 for that model).
- **S3 / AMP output recast**: `AMPWrapper` casts tensors back to FP32 recursively,
  preserving output structure (YOLOX tensor / YOLOv7 tuple / DAMO tuple|BoxList) so
  the imported adapters see baseline dtypes.
- **TRT / build from a FRESH baseline (not the optimized model)**: `optimize.py` calls
  `load_model()` again for the TRT stage. The earlier stages mutate the baseline in
  place (`fuse_conv_bn` turns every BN into `nn.Identity`; damo also wraps it in
  `torch.compile`), which would hide the `Conv→BN→activation` pattern TensorRT's
  matcher fuses. The reload is cheap (checkpoint already cached).
- **TRT / output-structure → mAP**: an engine returns raw tensors, not the baseline
  Python structure. YOLOX = single tensor → `yolox_adapter` works (`mAP_status="ok"`).
  DAMO standard export = single tensor → adapter also works (`mAP_status="ok"`).
  YOLOv7 (tuple) cannot be reconstructed → `mAP: null`,
  `mAP_status="skipped_output_format_mismatch"`. A genuine eval error →
  `failed_runtime_error`. Latency timing is unaffected (never inspects the output).
- **TRT / ONNX export quirks**: YOLOX and DAMO export cleanly. YOLOv7/x anchor decode
  in the `forward` is historically fragile to export but succeeded in the actual run
  (status=ok, latency measured); mAP is skipped due to tuple output format, not export
  failure. DAMO's standard ONNX export (with NMS in the graph) succeeded. A separate
  deploy-mode export (`trt_deploy_export.py`) bypasses NMS for a cleaner graph and
  achieves 6.04ms / 4.39×; its mAP is null (raw tensor, no adapter).
- **TRT / engine cache invalidation**: the serialized engine lives at
  `results/yolox_v7_damo/optimized/<model>/engine_fp16.trt` (torch_tensorrt path: `*.trt.ep`).
  It is reused only if it is **newer than that model's `load.py`** (passed as
  `invalidate_if_newer_than`); otherwise it is rebuilt. First build ~3–5 min, cached
  reuse ~10 s. Engine files are gitignored.
- **TRT / install = standalone `tensorrt` (NOT torch-tensorrt)**: the `tensorrt-pip`
  cell / `requirements-tensorrt.txt` install the standalone `tensorrt` (10.x) + `onnx`,
  which have **no torch dependency** — torch stays at the pinned `2.4.1`, the benches
  keep importing, and **no runtime restart is needed**. The wrapper still *tries*
  `torch_tensorrt.compile` first (it prints a harmless "unavailable" notice when that
  package is absent) and then builds via the ONNX → `tensorrt` API path, which is the
  only path that actually builds an engine for these models.
- **TRT / pin `tensorrt<11` — TRT 11 is strongly-typed-only**: an unbounded
  `pip install tensorrt` now resolves to **TensorRT 11.x** (Colab's newer base pulled
  `tensorrt 11.0.0.114`), which REMOVED the weakly-typed precision API the wrapper's
  ONNX path uses — `BuilderFlag.FP16`, `platform_has_fast_fp16`, and
  `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` are all gone (precision in TRT 11 comes
  from the ONNX dtypes, not a builder flag). The engine build then `AttributeError`s at
  `config.set_flag(trt.BuilderFlag.FP16)`. Fix: `requirements-tensorrt.txt` and both
  notebook pip cells pin **`tensorrt<11`** (`tensorrt>=10.0.0,<11`), restoring the tested
  weakly-typed FP16 path. `tensorrt_wrapper.py` also reads the flag via
  `getattr(trt.BuilderFlag, "FP16", None)` and raises a clear "pin `tensorrt<11`" error if
  a TRT 11+ wheel slips in — it never silently builds an FP32 engine mislabeled fp16.
  **If TRT 11 was already imported in the kernel** (e.g. you ran the smoke test once),
  re-running the pinned pip cell downgrades the wheel but you MUST **Runtime ▸ Restart
  session** before the in-kernel smoke test sees 10.x (the `run_all_optimizations`
  subprocess picks it up without a restart).
- **TRT / why we avoid `torch-tensorrt`** (`operator torchvision::nms does not exist`):
  `pip install torch-tensorrt==2.4.0` may pull a different `torch` (e.g. 2.4.0) than the
  project's `torch==2.4.1`, leaving `torchvision==0.19.1` ABI-mismatched, after which
  EVERY `bench.py` dies at `from torchvision.ops import nms` and only a runtime restart
  fixes it. We sidestep this entirely by installing the standalone `tensorrt` (torch
  untouched, no restart). If you ever install `torch-tensorrt` by hand and hit the nms
  error, recover with `!pip install -q torch==2.4.1 torchvision==0.19.1 --index-url
  https://download.pytorch.org/whl/cu121` then Runtime ▸ Restart session. The
  `tensorrt-pip` cell still lives at the END of notebook 05 so a top-to-bottom bench run
  never touches it.
- **Script path literals not yet updated after folder rename**: `run_all_optimizations.py`
  and `recompute_op_tables.py` still contain `results/yolo/optimized` and
  `OWNER = "jesussebastian"` (pre-rename literals). These must be updated to
  `results/yolox_v7_damo/optimized` and `OWNER = "yolox_v7_damo"` before a fresh Colab
  run, or the output CSVs will land in the wrong directories.

---

## Changes made in this project (vs. original scaffold)

1. **Real-COCO latency input** (all 6 `bench.py`): replaced `torch.randn()`
   with `ds[0][0].unsqueeze(0)` from `get_coco_val_subset_yolo()` — protocol
   now matches stated spec. Seeds fixed to 42 throughout.

2. **Fine-grained `_tag_protocol`** (notebook cell `063efe07`): Conv2d split
   by kernel size (Conv1x1/Conv3x3/Conv5x5/ConvNxN/DepthwConvNxN) + added
   LayerNorm, GroupNorm, GELU, Hardswish, AvgPool2d, Linear, MultiheadAttention,
   Upsample.

3. **Dynamic backbone_ops chart**: height scales with number of op types,
   tab20 palette, wider figure (12 in).

4. **Protocol cell** (`063efe07`): added `PREPROC` dict + real COCO input +
   seeds + `config.yaml` notes updated.

5. **S3 optimization phase** (`optim/yolox_v7_damo/`): new inference-only pipeline —
   Conv+BN fusion, AMP FP16, CUDA Graphs, and torch.compile, layered per model
   with stage-by-stage JSON + aggregated `optimization_summary.csv`. AMP FP16
   gives 1.22–2.56× across all six models with no mAP loss; CUDA Graphs (yolox_m)
   and torch.compile (yolov7/x) fall back gracefully when capture/trace fails.
   See the "Optimization phase (S3)" section above.

6. **TensorRT FP16 stage** (`optim/yolox_v7_damo/shared/tensorrt_wrapper.py` + new final stage
   in all 6 `optimize.py`): `TensorRTWrapper` exports a fresh FP32 baseline to ONNX
   (opset 17) and builds an FP16 engine (torch_tensorrt preferred, ONNX→`tensorrt`
   API fallback), with serialized-engine caching invalidated by `load.py` mtime.
   Stage JSON adds `engine_build_time_s`, `engine_cache_hit`, `status` and `mAP_status`
   (`ok` / `skipped_output_format_mismatch` / `failed_runtime_error`);
   `run_all_optimizations.py` gains a `status` column. **All 6 models succeeded** (3.62–4.31×
   over baseline). DAMO standard ONNX export worked (no NMS/decode block). YOLOv7/x
   exported cleanly; mAP skipped due to tuple output format only. A separate
   deploy-mode export for DAMO (`trt_deploy_export.py`) achieves 4.39× but has no
   mAP. **Install = standalone `tensorrt` (+ `onnx`), not `torch-tensorrt`**:
   it has no torch dependency, so the pinned `torch`/`torchvision` are left intact and
   **no runtime restart** is needed. New deps in `requirements-tensorrt.txt` /
   cell `tensorrt-pip`.

7. **Universal torch.compile stage** (`optim/yolox_v7_damo/shared/compile_util.py` + a
   `torch_compile` stage in all 6 `optimize.py`): every model now goes through one
   shared `attempt_compile` (mirrors Ultralytics' `attempt_compile`) so the stage is
   directly comparable across models. It tries `torch.compile(mode="reduce-overhead",
   backend="inductor")` (Inductor fusion **+ CUDA Graphs**) → `mode="default"`
   (fusion only) → eager, and records the landed path as **`compile_mode`** in the
   JSON. Added to yolox_m/l/x (previously had none); for yolov7/x the old
   default-backend attempt was replaced with this fallback chain; damo's two compile
   calls were routed through the same helper. yolox_m/l/x land on `reduce-overhead`
   (2.53–3.10×); yolov7/v7x fail all modes (AMP kept); damo lands `reduce-overhead`
   but gives no meaningful speedup at batch=1. `compile_util.py` is imported
   directly — `shared/__init__.py` stays frozen.

8. **Per-stage op-level profiler** (`optim/yolox_v7_damo/shared/stage_profiler.py`):
   runs a short `torch.profiler` pass immediately after each stage's latency benchmark
   and writes `stage_ops/<stage>_ops.csv` (op_type, mean_ms, pct_of_total) using
   the IDENTICAL `leaf_op_label` taxonomy as the notebook Protocol cell. Handles
   opaque stages: CUDA-graph replay and compiled graphs that suppress forward hooks
   → `fused_or_replayed_graph` placeholder; TensorRT engines → `profile_trt_stage_ops`
   attaches a `trt.IProfiler` to the execution context for real per-layer times,
   falling back to a `TRT_fused_engine` placeholder. `compute_speedup_factors()` joins
   all stages into `op_speedup_factors.csv` (Conv_total headline row + explicit sentinel
   cells: `fused_into_conv`, `opaque_compiled_graph`, `trt_fused_not_separable`,
   `below_threshold`, `stage_not_run`) + `op_speedup_factors_meta.csv` (per-stage
   profiling granularity). `stage_names.py` (torch-free) mirrors each optimize.py's
   `STAGE_NAMES` literal for CPU-only tooling.

9. **Sequential per-op latency table** (`op_latency_sequential.csv` per model +
   `op_latency_sequential_ALL.csv` stacked): Phase 2 output of `stage_profiler
   .build_sequential_latency_table` — per op, the mean_ms in every stage plus two
   factor families: `<stage>_factor_vs_baseline` and `<stage>_factor_vs_previous_stage`.
   Conv_total headline row first; all cells explicit (sentinel or number, never blank).
   Built without re-profiling by `recompute_op_tables.py`.

10. **CPU-only recompute entry point** (`optim/yolox_v7_damo/recompute_op_tables.py`):
    rebuilds `op_speedup_factors.csv`, `op_speedup_factors_meta.csv`, and
    `op_latency_sequential.csv` per model + `op_latency_sequential_ALL.csv` from
    existing `stage_ops` CSVs without a GPU or torch. Uses importlib to load
    `stage_profiler.py` and `stage_names.py` directly (bypasses the torch-dependent
    `shared/__init__.py`). Run: `python -m optim.yolox_v7_damo.recompute_op_tables`.
