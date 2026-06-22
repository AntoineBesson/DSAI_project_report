# CLAUDE.md — ProjectDSAI2026_2 (AOUIDET)

## Project context

Telecom Paris 2A SD / DSAI project. Profiling + inference-optimization suite for the
**single-stage YOLO-s family**, run in parallel with other student team 2 for cross-model comparison. Results must follow a shared
protocol so numbers are directly comparable across teams. Deployment target is
**ultra-low single-image (batch-1) latency** on a Tesla T4.

Owner tag: **AOUIDET Mohamed Yassine**
Branch: **yassine_yolo_trt_opt**
Repo: `gitlab.enst.fr:dsai/projects-2026/ProjectDSAI2026_2`
Runtime: Google Colab T4 GPU (16 GB GDDR6) — FP32 eager baseline, TensorRT FP16/INT8 for deployment

> **Scope note — two deliverables, two model sets.** The *profiling & optimization*
> phase covers **six** models (v5s, v6s, v8s, v9s, **v10s**, v11s). The final
> *TensorRT FP16 deployment* report covers **five** — **v10s is excluded from the TRT
> deployment deliverable per project constraints**. Keep this split in mind when
> regenerating tables: v10s appears in profiling/island roll-ups but NOT in the TRT
> engine tables.

> **YOLOv6s is latency-only.** It loads from a **random-initialised YOLOv6 YAML** —
> architecture and timing are valid, but the weights are untrained, so semantic mAP is
> not meaningful and there is **no FP32 baseline row**. All v6s speedups are reported
> relative to the island build, never an FP32 baseline.

---

## Models benchmarked

| Name | Architecture | Checkpoint source |
|---|---|---|
| `yolov5s` | YOLOv5-s (CSPDarknet + C3 blocks, anchor-based head) | Ultralytics YOLOv5 |
| `yolov6s` | YOLOv6-s (EfficientRep / RepVGG-style, **random-init YAML**) | Meituan YOLOv6 YAML — latency only |
| `yolov8s` | YOLOv8-s (C2f backbone, anchor-free DFL head) | Ultralytics YOLOv8 |
| `yolov9s` | YOLOv9-s (GELAN / RepNCSPELAN4 neck, PGI aux branches, DFL head) | Ultralytics YOLOv9 |
| `yolov10s` | YOLOv10-s (**NMS-free**, one-to-one `v10Detect` head) | Ultralytics / THU-MIG YOLOv10 |
| `yolov11s` | YOLO11-s (C2PSA attention backbone, DFL head) | Ultralytics YOLO11 |

Architectural hotspots flagged in the diagnosis: the **detection head** is the heaviest
brick on every NMS-bearing model, and the **stacked RepNCSPELAN4 backbone/neck** is the
heaviest on v9s.

---

## Benchmark protocol (exact, cross-team comparable)

- **GPU**: NVIDIA Tesla T4, FP32 eager mode for the baseline (no torch.compile / TensorRT / quantization)
- **Input**: single **640×640** image, index-0 of the seed=42 **2000-image COCO val2017** subset, letterboxed; YOLO-s preprocessing is RGB → `[0,1]` (÷255)
- **Latency**: **50 warmup + 1000 timed** `model(x)` calls, each bracketed by `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()`
- **Seeds**: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)` at the top of every `main()`
- **Post-processing**: decode/NMS included in the measured forward pass (it is part of the host-bound cost the optimization targets)
- **mAP**: COCO val2017 **2000-image** subset (seed 42), mAP@50–95 — small inter-stage deltas are within subset noise and should read as "preserved," not as gains

---

## Optimization pipeline (deployment stack)

The single largest lever on this hardware is **not quantization** — every profiled model
is **host-bound** (61–82% CPU/Python dispatch, only ~12–26% real GPU compute, all
convolutions FP32 / no Tensor-Core use). The dominant win comes from removing per-op
Python dispatch via a captured engine; quantization then shrinks the ~15% GPU slice.

**Shared optimization stack (apply to all, in this order):**

1. **Export** PyTorch → ONNX (fold reparam blocks, strip training-only branches, e.g. PGI on v9s)
2. **Build TensorRT FP16 engine** → validate mAP
3. **INT8 PTQ** with calibration as a mixed engine (keep DFL head & attention in FP16) — *planned next stage*
4. **EfficientNMS** folded into the engine (static output) for the NMS-bearing models — *planned next stage*
5. **CUDA graph** capture to erase the 61–82% host-dispatch overhead

**PyTorch-side fallback — the "island build"** (deployable when a TRT engine cannot be
built; also the diagnostic that locates where latency lives): per-brick `conv_bn_fold`
(lossless) + selective **FP16 Tensor-Core islands** (FP16 only on the convs that
measurably benefit). TensorRT internalises both levers — it folds Conv+BN, runs FP16
kernels everywhere they help, and fuses Conv+BN+activation into single kernels — which is
why the full engine beats the hand-tuned island build by a further **1.25×–2.54×**.

---

## Key files

> Representative layout for this pipeline — align paths with the actual repo tree on
> `yassine_trt_opt`. The namespace mirrors the cross-team convention
> (`<owner-family>/…`); here the family is **`yolo_s`**.

### Benchmark scripts (FP32 baseline, protocol-correct)
```
models/yolo_s/<model>/bench.py        # 6 scripts — latency + mAP, one per model
models/yolo_s/run_all_baselines.py    # runs all 6 sequentially → baseline_yolo_s.csv
models/yolo_s/_yolo_coco_dataset.py   # CocoYoloDataset, get_coco_val_subset_yolo(), letterbox
```

### Model loaders
```
models/yolo_s/<model>/load.py         # load_model() — downloads checkpoint on first run
models/yolo_s/yolov6s/load.py         # builds from random-init YOLOv6 YAML (latency only)
models/yolo_s/yolov9s/load.py         # strips PGI aux branches, fuses RepConv before export
```

### Optimization scripts (TRT phase — inference-only)
```
optim/yolo_s/shared/conv_bn_fold.py        # fuse_conv_bn() — lossless, per-fold losslessness check
optim/yolo_s/shared/fp16_islands.py        # selective FP16 Tensor-Core cast (per-brick island selection)
optim/yolo_s/shared/tensorrt_wrapper.py    # ONNX(opset17) → TRT FP16 engine, FP32 in/out, engine caching
optim/yolo_s/shared/cuda_graph_wrapper.py  # capture/replay around the built engine
optim/yolo_s/shared/efficient_nms.py       # EfficientNMS plugin folding (planned, NMS-bearing models)
optim/yolo_s/shared/int8_ptq.py            # INT8 calibration / mixed-precision engine (planned)
optim/yolo_s/shared/brick_profiler.py      # per-brick lever sweep (fold / fp16 / fold+fp16) → factors CSVs
optim/yolo_s/<model>/optimize.py           # per-model pipeline (stages → JSON)
optim/yolo_s/run_all_optimizations.py      # runs all, writes optimization_summary.csv + roll-ups
```

### Main notebook
```
utils/notebooks/05_run_yolo_s_benches.ipynb
```
Key regions: pip/setup cells → baseline `run_all_baselines` (Option B) → **nsys cell**
(host/GPU split, run before any torch.profiler cell) → **Protocol cell** (pure-FP32
deliverable: latency + torch.profiler breakdown + charts + CSVs) → **TRT cell** (at the
**end**: installs the **standalone `tensorrt` wheel**, no torch dependency, **no runtime
restart**) → zip + Drive backup.

---

## Deliverable tree (per model)

```
outputs/yolo_s/<model>/
├── config.yaml
├── results/
│   ├── latencies.csv               # raw 1000-iter timings
│   ├── summary.csv                 # mean/std/p50/p95/p99/fps/params
│   ├── submodules.csv              # backbone/neck/head % breakdown
│   ├── system_breakdown.csv        # gpu_compute / cpu_launch / post_proc / wall
│   ├── brick_levers.csv            # per-brick fold / fp16 / fold+fp16 factors
│   ├── stage_progression.csv       # FP32 → island build → TRT FP16 p50
│   └── trt_engine.csv              # p50/mean/p99/std/fps + load_ms + device_mem_mb
├── traces/
│   └── <model>_trace.json          # chrome://tracing compatible
├── nvidia_profiles/
│   ├── nsys_benchmark.nsys-rep
│   └── nsys_stats.txt
└── charts/
    ├── stacked_latency.png                 # backbone/neck/head stacked bar
    ├── winning_lever_per_brick.png         # conv_bn_fold / fold+fp16 / fp16 stacked
    ├── median_fp32_vs_opt.png              # p50 baseline vs optimized
    ├── tail_p99_log.png                    # p99 jitter collapse (log scale)
    └── accuracy_vs_latency_pareto.png      # mAP vs optimized median latency
```

---

## How to run in a new Colab session (baseline + protocol)

1. Run `00_setup_colab.ipynb` — mounts Drive, clones repo, installs deps (~3 min)
2. In `05_run_yolo_s_benches.ipynb`: run pip + setup cells
3. **Optional:** run **Option B** — `!python -m models.yolo_s.run_all_baselines` (~30 min, produces FP32 baseline mAP)
4. Run **Protocol cell** — ~10–15 min, generates all FP32 deliverables (latency, charts, traces)
5. Run **Zip / Drive backup cell** to download

To also include TensorRT results: run the separate TRT phase below (after the protocol is done).

---

## Optimization phase — stages & per-model map

Inference-only optimizations layered on top of the FP32 baselines. Each model has its own
`optimize.py` that benchmarks the baseline, then each stage, writes per-stage JSON, and
measures mAP once on the fully optimized engine. `run_all_optimizations.py` aggregates
all into `optimization_summary.csv`.

### Per-model optimization map

| Model | Conv-BN fold | FP16 islands | Island build | TensorRT FP16 | CUDA graph | INT8 + EfficientNMS (planned) | mAP measured |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| yolov5s | ✅ | ⚠️ regresses on C3¹ | ✅ (fold-led) | ✅ | ✅ | INT8 + EffNMS | ✅ |
| yolov6s | ✅ | ✅ **backbone champ**² | ✅ | ✅ | ✅ | fold+fp16 → INT8 | ❌ random-init |
| yolov8s | ✅ | partial | ✅ | ✅ | ✅ | INT8 (DFL FP16) + EffNMS | ✅ |
| yolov9s | ✅ | partial (neck resists)³ | ✅ | ✅ | ✅ | reparam-GELAN → INT8 + EffNMS | ✅ |
| yolov10s | ✅ | partial | ✅ | **excluded from TRT report** | — | INT8 end-to-end (NMS-free) | profiling only |
| yolov11s | ✅ | ✅ (C2PSA fold) | ✅ | ✅ | ✅ | INT8 conv + attn/DFL FP16 + EffNMS | ✅ |

¹ v5s: oldest C3 recipe — FP16 frequently **regresses** at the brick level (small batch-1
tiles under-fill the Tensor Cores), so fold alone is selected almost everywhere and v5s
leans on launch removal, not precision.
² v6s: the **only** model where FP16 truly carries the backbone (2.13× backbone,
fold+fp16 best lever at 2.49×).
³ v9s: the heavy GELAN / RepNCSPELAN4 neck resists a blanket FP16 cast, so the island
build moves it only **1.13×** — yet TensorRT fuses the neck end-to-end and lands **2.88×**.
This is the most instructive single finding: the model the island build could barely move
is the one the engine accelerates most.

---

## Measured results (T4, batch = 1)

### Final result per model — TensorRT FP16 (the deployment winner)

Speedup = FP32-baseline p50 ÷ TRT-FP16 p50. **TRT FP16 is the best stage on all five models.**

| Model | FP32 p50 (ms) | TRT FP16 p50 (ms) | Speedup | FPS | mAP (subset 2000) |
|---|--:|--:|--:|--:|--:|
| yolov5s | 13.00 | 7.32 | 1.78× | 136.7 | 0.438 |
| yolov6s | n/a¹ | 5.59 | n/a¹ | 179.0 | n/a¹ |
| yolov8s | 14.01 | 7.53 | 1.86× | 132.8 | 0.460 |
| yolov9s | 25.05 | 8.69 | 2.88× | 115.0 | 0.476 |
| yolov11s | 17.27 | 7.55 | 2.29× | 132.4 | 0.477 |

¹ v6s: random-init YAML — latency/throughput only, no FP32 baseline, no semantic mAP.

Speedup grows with the size of the FP32 baseline (v5s/v8s ≈13–14 ms → ≈1.8×; v11s 17 ms →
2.29×; v9s 25 ms → 2.88×). v6s tops throughput at **179 FPS** as a backbone-bound model
(almost all plain convs TensorRT fuses onto Tensor Cores). mAP lands at or just above the
FP32 subset value on every trained model (v5s 0.425→0.438, v8s 0.444→0.460, v9s
0.460→0.476, v11s 0.464→0.477) — within subset noise, no FP16 accuracy loss.

### Stage-by-stage progression (median p50, ms)

| Model | FP32 baseline | FP16 islands + fold | TensorRT FP16 | TRT vs baseline | TRT vs islands |
|---|--:|--:|--:|--:|--:|
| yolov5s | 13.00 | 9.18 | 7.32 | 1.78× | 1.25× |
| yolov6s | — | 11.43 | 5.59 | — | 2.05× |
| yolov8s | 14.01 | 9.80 | 7.53 | 1.86× | 1.30× |
| yolov9s | 25.05 | 22.12 | 8.69 | 2.88× | 2.54× |
| yolov11s | 17.27 | 11.37 | 7.55 | 2.29× | 1.51× |

The island build captures part of the gain but leaves a residual only the engine reaches —
biggest on v9s (island 1.13× → TRT 2.54× on top of it), smallest on v5s (1.25×, fold-led).

### TensorRT engine — latency & throughput (single-shot, no graph capture)

| Model | p50 (ms) | mean (ms) | p99 (ms) | std (ms) | FPS |
|---|--:|--:|--:|--:|--:|
| yolov5s | 7.318 | 7.332 | 7.56 | 0.093 | 136.7 |
| yolov6s | 5.586 | 5.601 | 5.78 | 0.059 | 179.0 |
| yolov8s | 7.531 | 7.539 | 7.73 | 0.057 | 132.8 |
| yolov9s | 8.692 | 8.691 | 8.91 | 0.096 | 115.0 |
| yolov11s | 7.552 | 7.570 | 7.79 | 0.107 | 132.4 |

Mean ≈ median to within 0.02 ms; p99 sits only 0.2–0.4 ms above p50 — the signature of a
static execution graph. All engines load in **< 65 ms** and occupy **≈900–940 MB** device
memory (v5s 902 MB / 20 ms, v6s 938 MB / 40 ms, v8s 910 MB / 27 ms, v9s 912 MB / 64 ms,
v11s 915 MB / 52 ms). GPU utilisation is **100%** across the board.

### The tail is the real win (p99 & jitter)

| Model | FP32 p99 (ms) | TRT p99 (ms) | p99 reduction | FP32 std (ms) | TRT std (ms) |
|---|--:|--:|--:|--:|--:|
| yolov5s | 600.8 | 7.56 | 79× | 74.7 | 0.093 |
| yolov8s | 554.4 | 7.73 | 72× | 70.4 | 0.057 |
| yolov9s | 395.3 | 8.91 | 44× | 45.4 | 0.096 |
| yolov11s | 388.8 | 7.79 | 50× | 46.9 | 0.107 |

FP32 baselines spent 60–82% of wall-time in CPU/Python dispatch, so their p99 reached
0.4–0.6 s and std reached 45–75 ms — dispatch jitter, not compute. The engine executes as
a single static dispatch of fused kernels, collapsing the tail to median + sub-0.1 ms
jitter. v6s carries no FP32 baseline (no tail reduction computed); its engine std of
**0.059 ms is the lowest of the set** — the most deterministic engine. For real-time
single-image detection, this determinism — not the median — is what matters.

### CUDA graph capture (amortised throughput + tighter jitter)

| Model | Engine p50 (ms) | +Graph p50 (ms) | Engine pipelined (ms/img) | +Graph pipelined (ms/img) | +Graph std (ms) |
|---|--:|--:|--:|--:|--:|
| yolov5s | 7.318 | 7.381 | 7.40 | 7.32 | 0.094 |
| yolov8s | 7.531 | 7.552 | 7.54 | 7.56 | 0.049 |
| yolov9s | 8.692 | 8.870 | 8.80 | 8.77 | 0.128 |
| yolov11s | 7.552 | 7.670 | 7.65 | 7.59 | 0.067 |

v6s: engine p50 5.586 → graph 5.744 ms; pipelined 5.65 → 5.73 ms/img.

Standalone p50 is flat-to-marginally-higher under capture (replay bookkeeping the
single-shot timer exposes while compute dominates). The benefit shows in the **pipelined**
metric (v5s 7.40→7.32, v9s 8.80→8.77, v11s 7.65→7.59) and in jitter (v8s 0.057→0.049,
v11s 0.107→0.067). The graph variant is where mAP was measured; capture changes only
dispatch, not engine math, so those mAP values apply equally to the non-graph engine.

### Island build — per-stage roll-up (PyTorch-side fallback)

| Model | Backbone (×) | Neck (×) | Head (×) | FP32 sum (ms) | Islands sum (ms) |
|---|--:|--:|--:|--:|--:|
| yolov5s | 1.42 | 1.35 | 1.34 | 13.86 | 10.01 |
| yolov6s | 2.13 | 1.22 | 1.02 | 11.74 | 7.02 |
| yolov8s | 1.22 | 1.08 | 1.02 | 9.84 | 8.67 |
| yolov9s | 2.02 | 1.95 | 1.28 | 28.21 | 14.78 |
| yolov10s | 1.34 | 1.41 | 2.48 | 14.19 | 8.71 |
| yolov11s | 1.44 | 1.44 | 1.72 | 12.26 | 8.20 |

v9s is **neck-bound**: its GELAN/RepNCSPELAN4 neck alone drops 10.6 → 5.4 ms (1.95×), the
single biggest absolute saving of the family. v6s is **backbone-bound** (2.13×, FP16
carries it). v8s bricks are already lean (most modest 1.0–1.2× stage gains).

### Dominant lever per model (per-brick analysis)

| Model | Bricks | Dominant lever | Heaviest brick (stage) | Best lever (factor) |
|---|--:|---|---|---|
| yolov5s | 25 | conv_bn_fold | Detect (head, 2.97 ms) | conv_bn_fold (1.34×) |
| yolov6s | 29 | conv_bn_fold | Sequential (backbone, 1.98 ms) | fold+fp16 (2.49×) |
| yolov8s | 23 | conv_bn_fold | Detect (head, 2.16 ms) | conv_bn_fold (1.01×) |
| yolov9s | 23 | conv_bn_fold | RepNCSPELAN4 (backbone, 3.34 ms) | conv_bn_fold (2.03×) |
| yolov10s | 24 | conv_bn_fold | v10Detect (head, 5.26 ms) | conv_bn_fold (2.48×) |
| yolov11s | 24 | conv_bn_fold | Detect (head, 2.65 ms) | conv_bn_fold (1.73×) |

Conv-BN fold is the **universally safe lever** — it wins on the large aggregation bricks
(C3 / C2f / RepNCSPELAN4 / C2PSA / PSA) where folding the reparameterizable structure
deletes whole kernel launches at batch 1. FP16 islands pay off on the plain convs of the
newer backbones (v6/v8/v11: fold+fp16 reaches 1.4×–3.4×). TensorRT internalises both
levers everywhere they help, which is why the engine beats the hand-selected island build
by another 1.25×–2.54×.

### Profiling — host-bound diagnosis (nsys, FP32 baseline)

| Model | median ms | mean ms | p99 ms | std ms | GPU % | CPU/Py % | post-proc % | mAP@50-95 | data |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| yolov5s | 13.00 | 23.30 | 600.8 | 74.7 | 26.4 | 62.8 | 13.5 | 0.4252 | full profile |
| yolov8s | 14.01 | 23.81 | 554.4 | 70.4 | 17.2 | 78.3 | 25.0 | 0.4443 | full profile |
| yolov9s | 25.05 | 32.96 | 395.3 | 45.4 | 22.9 | 60.7 | 6.5 | 0.4601 | full profile |
| yolov10s | 15.48 | 17.37 | 44.9 | 8.07 | 12.4 | 82.2 | NMS-free | 0.4563 | summary + sys |
| yolov11s | 17.27 | 24.31 | 388.8 | 46.9 | 14.7 | 79.9 | 25.6 | 0.4635 | full profile |

Every model is host-bound (61–82% CPU/Python, only ~12–26% GPU compute). NMS-bearing
models (v5/v8/v9/v11) carry a 13–26% post-process cost that, combined with host jitter,
drives a catastrophic latency tail (p99 up to 601 ms). The **NMS-free v10s already shows
the best tail by far** (p99 45 ms, std 8). v10s is the only model still lacking op-level /
per-stage detail.

### Island build — optimized latency & tail (profiling phase)

| Model | FP32 p50 (ms) | Opt p50 (ms) | Speedup | Opt FPS | FP32 p99 (ms) | Opt p99 (ms) | p99 ↓ | Opt std (ms) | mAP FP32→Opt |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| yolov5s | 13.00 | 9.18 | 1.42× | 109.0 | 600.8 | 18.2 | 33× | 2.42 | 0.425→0.434 |
| yolov6s | — | 11.43 | — | 87.5 | — | 14.4 | — | 0.62 | NA |
| yolov8s | 14.01 | 9.80 | 1.43× | 102.0 | 554.4 | 17.2 | 32× | 1.97 | 0.444→0.452 |
| yolov9s | 25.05 | 22.12 | 1.13× | 45.2 | 395.3 | 42.3 | 9× | 4.91 | 0.460→0.469 |
| yolov10s | 15.48 | 11.29 | 1.37× | 88.5 | 44.9 | 21.4 | 2× | 2.62 | 0.456→0.469 |
| yolov11s | 17.27 | 11.37 | 1.52× | 88.0 | 388.8 | 22.6 | 17× | 2.79 | 0.464→0.469 |

Median improves a moderate 1.1–1.5×, but **p99 collapses up to 33×** (v5s 601→18 ms) and
run-to-run std up to 36× (v8s 70.4→1.97 ms). These are island-build (PyTorch fallback)
mAP values; the TRT engine reaches slightly higher (v5s 0.438, v8s 0.460, v9s 0.476, v11s
0.477) — see the final-result table.

### Brick-model projection vs measured (Amdahl roll-up = honest headroom)

| Model | Projected total (ms) | Projected speedup | Measured opt p50 (ms) | Measured speedup |
|---|--:|--:|--:|--:|
| yolov5s | 10.01 | 1.38× | 9.18 | 1.42× |
| yolov6s | 7.02 | 1.67× | 11.43 | — |
| yolov8s | 8.67 | 1.14× | 9.80 | 1.43× |
| yolov9s | 14.78 | 1.91× | 22.12 | 1.13× |
| yolov10s | 8.71 | 1.63× | 11.29 | 1.37× |
| yolov11s | 8.20 | 1.50× | 11.37 | 1.52× |

The projection sums best per-brick levers and ignores residual host overhead / inter-brick
stalls, so it is optimistic. For **v9s the projection (1.91×) far exceeds the measured
island 1.13×** — the heavy GELAN neck is the prime candidate for further work (reparam
fusion + FP16 on the neck convs), which TensorRT then realises (2.88×).

---

## Per-model recommended build & reasoning

Each recommendation combines measured nsys/PyTorch profiles with architecture-level reasoning.

**yolov5s** → TensorRT **INT8** (mixed FP16 islands) + Conv-BN fold + EfficientNMS + CUDA graph
- Least host-bound of the set (62.8% CPU/Python vs 26.4% GPU), high launch overhead → CUDA graph still the top lever, larger GPU share makes FP16/INT8 pay off.
- Conv2d = 90% of GPU, all FP32 → FP16 engages Tensor Cores (~2× on convs). Lowest post-process (13.5%) yet **worst tail (p99 601 ms)** = host/sync jitter, flattened by graph capture.
- Lowest mAP (0.4252, oldest arch), no reparam blocks → cleanest Conv-BN fold and the most mature INT8 recipes; INT8 can be pushed hardest with least relative loss.

**yolov8s** → TensorRT **FP16→INT8** (DFL kept FP16) + EfficientNMS + CUDA graph
- 78.3% CPU/Python → host-bound; CUDA graph + static execution is the #1 lever, ahead of quantization.
- NMS = 25% drives the p99 554 ms tail → GPU-side EfficientNMS removes the device↔host sync stall. C2f emits ~10,500 tiny Reshape/View ops/iter → TRT fusion collapses them; keep DFL in FP16 to protect mAP (0.4443).

**yolov9s** → reparam-fused GELAN → TensorRT **INT8** (DFL FP16) + EfficientNMS + CUDA graph
- Fires ~37,200 Conv2d ops/iter (2–3× the others) + ~22,709 tiny Reshape/View → record 16.2% launch overhead; CUDA-graph capture pays off most here.
- Slowest model (25 ms, 39.9 FPS); time is lopsided into backbone 48.9% + neck 32.4% (81% combined), post-process only 6.5% — the cost is GELAN aggregation, not NMS. Fuse RepConv into single convs before export, strip PGI aux branches. mAP 0.4601.

**yolov10s** → TensorRT **INT8 end-to-end** (NMS-free) + single full-pipeline CUDA graph **★ best fit for the goal**
- NMS-free (one-to-one head at inference) → no post-process sync; already the best measured tail (p99 45 ms, std 8) and best accuracy (0.4563). Static end-to-end → the whole pipeline fits one CUDA graph / one engine.
- Most host-bound of all (82.2% CPU/Python) → graph capture is the main unlock; no DFL/NMS quirk → INT8 can be applied most aggressively. *(Excluded from the current TRT deployment report, but prioritized for the goal.)*

**yolov11s** → TensorRT **INT8** conv backbone + C2PSA attention & DFL kept FP16 + EfficientNMS + CUDA graph
- Most host-bound (79.9% CPU/Python) → CUDA graph + static execution is decisive. C2PSA attention (124 µs/iter, FP32 + Linear projections) is INT8-sensitive → keep attention + DFL in FP16, quantize only the conv backbone/neck. NMS retained (post-process 25.6%, p99 389 ms) → EfficientNMS folds it in. Best measured island median (1.52×), mAP 0.4635.

**Expected ordering after full optimization (batch-1 latency):** v10s fastest & lowest-jitter
(NMS-free + single static graph) > v5s / v8s / v9s (NMS folded, FP16/INT8 convs) > v11s
heaviest (attention block + retained NMS). Biggest single win across all: the CUDA-graph /
TRT capture that removes the ~80% Python-dispatch overhead; FP16 roughly halves conv time;
mixed-INT8 adds a further ~1.3–1.7× on quantized convs on T4.

---

## Per-model preprocessing (for `get_coco_val_subset_yolo`)

| Model | `channel_order` | `scale` |
|---|---|---|
| yolov5s / v8s / v9s / v10s / v11s | `"rgb"` | `1/255` (→ [0,1]) |
| yolov6s | `"rgb"` | `1/255` (random-init — latency only) |

All YOLO-s models letterbox to 640×640 (gray pad), RGB, normalized to `[0,1]`.

---

## Known issues / caveats

- **v10s is excluded from the TRT deployment deliverable** (project constraint). It is
  fully present in profiling / island roll-ups but absent from every TensorRT engine
  table. Do not add a v10s row when regenerating the TRT FP16 tables.
- **YOLOv6s is latency-only** — random-init YOLOv6 YAML. Architecture/timing valid,
  semantic mAP not meaningful, no FP32 baseline. All v6s speedups are vs the island
  build, never an FP32 baseline.
- **TensorRT has no fine-grained op breakdown.** An engine runs as fused myelin kernels,
  so a profiler sees one Conv bucket + a small residual rather than separate
  Conv/activation/pool ops. The per-brick factors therefore come from the **PyTorch
  island build**, not the engine — the engine's lack of a per-op split is itself evidence
  of successful fusion.
- **CUDA-graph p50 is marginally higher than the bare engine at batch 1** (capture/replay
  bookkeeping). The graph's value is in **pipelined throughput and jitter**, not
  single-shot median; both variants share the same engine and the same mAP.
- **mAP is a 2000-image subset** (seed 42), not the full 5000-image val set. Small
  inter-stage deltas are within subset noise — read as "preserved," not as genuine gains.
- **FP32 baselines were host-bound** (60–82% of wall-time in CPU/Python dispatch). Their
  giant p99/std came from dispatch jitter, which is why the tail reductions are so large —
  they measure determinism recovered, not raw compute saved.
- **FP16 regresses on v5s C3 blocks** — small batch-1 tiles under-fill the Tensor Cores
  (`f_fp16 < 1`). Fold alone is selected almost everywhere for v5s; do not force a blanket
  FP16 cast on it.
- **Conv-BN fold is lossless and subsumed by the engine.** A per-fold losslessness check
  guards it (passed on all reported models). TensorRT performs the same
  Conv+BN+activation fusion automatically, so the standalone PyTorch fold is a strict
  subset of what the engine does.
- **TRT install = standalone `tensorrt` wheel (NOT `torch-tensorrt`)** — it has no torch
  dependency, so the pinned torch/torchvision stay intact and **no runtime restart** is
  needed. **`requirements.txt` MUST pin `tensorrt>=10.0,<11`** — the weakly-typed FP16
  builder-flag path needs it. An unpinned `tensorrt>=10.0` resolves to **TRT 11 on current
  Colab**, which is strongly-typed-only and removes `BuilderFlag.FP16`, raising
  `AttributeError: type object '…BuilderFlag' has no attribute 'FP16'` at
  `cfg.set_flag(trt.BuilderFlag.FP16)`. With the pin in place the original `trt_build.py`
  builds unmodified — no `BuilderFlag` rename shim or in-notebook `txt.replace` patch is
  needed. (If the printed `tensorrt.__version__` is still `11.x` because Colab preinstalled
  and imported it, do Runtime → Restart **once** and re-run from the pip cell.) Keep the TRT
  pip cell at the **end** of notebook 05 so a top-to-bottom baseline run never touches it.
- **Colab disk is ephemeral** — `/content/` is wiped on disconnect; only Google Drive
  persists. Save to Drive after every run. Engine files (`*.trt` / `*.engine` / `*.onnx`)
  are gitignored and excluded from the deliverable zip (otherwise ~1 GB bloat).
- **Set cwd with `os.chdir`, not just `%cd`.** `config.py` creates `artifacts/onnx`,
  `artifacts/engines`, `artifacts/caches` at **import time** from relative paths, so the
  process working directory must already be the project root when `yolo_trt` is first
  imported. After unzipping, call `import os; os.chdir("/content/yolo_trt_optim")` before any
  `yolo_trt` import — otherwise the dirs are created under `/content` and
  `torch.onnx.export` raises `FileNotFoundError: [Errno 2] … 'artifacts/onnx/<model>.onnx'`.
- **Each TRT/benchmark cell imports what it uses.** Import `build_tensorrt_engine`,
  `benchmark_engine`, `validate_accuracy` at the top of the cell that calls them rather than
  relying on a single earlier import cell. A runtime restart (or running cells out of order)
  otherwise yields `NameError: name 'benchmark_engine' is not defined` even though `engine`
  is still bound. Also delete any orphan test cell that references `onnx_path` before
  `export_to_onnx()` assigns it (`NameError: name 'onnx_path' is not defined`).
- **nsys CUPTI conflicts** — nsys profiling fails intermittently on Colab if
  `torch.profiler` ran first in the same kernel. Run the nsys cell **before** the protocol
  cell on a fresh kernel.

---

## Changes made in this project (vs. original scaffold)

1. **Real-COCO latency input** (all `bench.py`): replaced `torch.randn()` with the seed=42
   2000-image COCO val2017 subset index-0 image; seeds fixed to 42 throughout, matching
   the cross-team protocol.

2. **Per-brick lever sweep** (`brick_profiler.py`): every model decomposed brick-by-brick
   (indexed backbone/neck/head sub-modules); three levers measured per brick
   (`conv_bn_fold`, `fp16`, `fold+fp16`) and the best kept → `brick_levers.csv` +
   `winning_lever_per_brick.png`. Per-stage roll-up to backbone/neck/head factors.

3. **Host-bound diagnosis** (nsys): GPU vs CPU/Python vs post-process split for all six
   models — the central finding that the headline gain is dispatch-tail collapse, not raw
   compute. Identified v9s as neck-bound, v6s as backbone-bound, v10s as the
   already-deterministic NMS-free baseline.

4. **Island build** (`conv_bn_fold.py` + `fp16_islands.py`): PyTorch-side fallback —
   lossless Conv-BN folding + selective FP16 Tensor-Core islands, layered per model. Gives
   1.13×–1.52× over baseline with no mAP loss, and 9×–33× p99 / up to 36× std collapse;
   serves as the deployable fallback wherever a TRT engine cannot be built.

5. **TensorRT FP16 stage** (`tensorrt_wrapper.py` + final stage in all five engine
   models): exports a fresh FP32 baseline to ONNX (opset 17) and builds an FP16 engine with
   serialized-engine caching invalidated by `load.py` mtime. **All five engine models
   succeeded** — 1.78×–2.88× over the FP32 baseline at the median and, more importantly,
   **44×–80× lower at p99** with run-to-run std under 0.11 ms. Engines run at 100% GPU
   utilisation. The most instructive finding: **v9s** (island build 1.13×) is the model the
   engine accelerates most (2.88×), because its reparameterizable neck is exactly the
   structure TensorRT fusion was built to collapse.

6. **CUDA graph capture** stage on the built engine: flat-to-marginally-higher single-shot
   p50 but improved pipelined throughput (v5s 7.40→7.32 ms/img) and tighter jitter (v8s
   0.057→0.049, v11s 0.107→0.067). mAP measured on the graph variant applies equally to
   the non-graph engine.

7. **Per-model deployment recommendation** (recommended optimized build per model):
   combined measured profiles + architecture reasoning into a per-model INT8 + EfficientNMS
   + CUDA-graph stack, with v10s (NMS-free, end-to-end INT8, single full-pipeline graph)
   flagged as **★ best fit** for the ultra-low-latency goal.

8. **Next stage (planned)** — INT8 PTQ with calibration (mixed engines: DFL head &
   attention kept FP16) + EfficientNMS folded into the engine for the NMS-bearing models,
   plus op-level / per-stage profiling for v10s to size its FP16/INT8 gains layer-by-layer
   (the only remaining profiling gap).
