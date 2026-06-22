# DSAI Project — Chat History So Far

> Generated from the project conversation context available in this ChatGPT project. Some earlier conversations were partially truncated in the available context, so this file records the accessible titles, dates, user requests, decisions, and outcomes rather than a verbatim full transcript.

## Project context

The project concerns profiling and optimising object detection models on COCO, with a particular focus on the P6 assignment covering DINO, Co-DETR, and RT-DETR R18/R50/R101. Across the chats, the work converged toward RT-DETR benchmarking, profiling, report writing, and acceleration analysis using PyTorch eager, `torch.compile`, AMP FP16, ONNX Runtime, TensorRT, Nsight Systems, and kernel-level analysis.

Important uploaded/context files included:

- `Email-from-Professor.txt`
- `orga début project dsai.pdf`
- `pasted.txt`
- `requirements.txt`
- `Code Profiling Feedback.txt`
- `Profiling-Results.txt`
- `Email-from-Professor-2.txt`
- `protocole_commun_benchmark_profiling.pdf`
- `Email-from-Professor-3.txt`
- `rapport_dsai_project.pdf`
- `rapport_dsai_project_vansh.pdf`
- `final_results.zip`
- `ProjectDSAI2026_2.zip`
- `ORT_fp16_results_nsys_only.zip`
- `ORT_fp16_results_pyprofile.zip`

---

## Conversation 1 — Model Download Process

**Date:** 2026-06-20  
**Main theme:** Code organisation and project restructuring.

### Key discussion points

- Asked whether models are re-downloaded every time the code runs.
- Asked how the code should be organised according to a project layout such as:

```text
models/nom_famille
optim/nom_famille
output/nom_famille
utils/nom_famille
```

- Uploaded the most up-to-date code as `ProjectDSAI2026_2.zip`.
- Requested code reorganisation under the family name `rt-detr`.
- Required all runnable code, bash scripts, requirements, README, and notebook to be placed under:

```text
utils/rt-detr
```

- Required the functionality to remain unchanged despite folder renaming and restructuring.
- Later asked whether the remaining `benchmark` folder was still useful or redundant.

### Outcome / direction

The intended structure became family-based, with RT-DETR as the family name. The focus was on minimal code changes, preserving existing behaviour, and updating run instructions so the reorganised project could still execute as before.

---

## Conversation 2 — Code protocole COCO

**Date:** 2026-05-27  
**Main theme:** Benchmark protocol, source patches, and Nsight script correction.

### Key discussion points

- Shared benchmark logs showing RT-DETR Hugging Face models being run:

```text
models=['rtdetr_hf_r18', 'rtdetr_hf_r50', 'rtdetr_hf_r101']
output=outputs_quick/P6
```

- Asked for an actual patch in code form rather than a diff file.
- Clarified that the desired answer was direct replacement code to paste into the source files.
- Observed that the Nsight bash script did not run the model on the dataset according to the protocol; it seemed to run a simpler benchmark.
- Asked for the script to be updated so it runs the model the same way as the full run script, but under Nsight.
- Asked whether running that script gives all the data from `run_full_hf_on_t4` plus Nsight.
- Reported an Nsight error:

```text
==ERROR== unrecognised option '--output'. Use --help for further details.
```

### Outcome / direction

The chat focused on making the profiling scripts protocol-compliant: same COCO subset, batch size 1, same model path, same evaluation pipeline, and adding Nsight capture rather than replacing the real benchmark with a toy run.

---

## Conversation 3 — Convolution Kernel Comparison

**Date:** 2026-06-15  
**Main theme:** Writing optimisation-analysis sections for the report.

### Key discussion points

- Asked why count per iteration was used when the number of iterations was not the same across profiles.
- Requested similar analysis sections for:
  - GEMM operations.
  - Elementwise kernels.
  - Kernel fusion analysis.
  - Launch overhead reduction.
- Asked whether RT-DETR even has 1x1 convolutions, and whether that removes ambiguity in the convolution analysis.
- Asked for other important subsections to include in the acceleration analysis.
- Requested LaTeX code for a terse subsection on kernel fusion analysis, with at most one table.
- Asked for more concrete examples comparing individual PyTorch eager operations with fused ORT/TensorRT operations.
- Asked for the analysis to use extracted metrics and be placed directly in LaTeX.
- Requested a single paragraph explaining the operator-level results using RT-DETR R50 as baseline, covering the share of time taken by convolution, GEMM, launch overhead, and post-processing in PyTorch eager.

### Outcome / direction

The chat produced report-ready analysis framing the acceleration story around concrete operator classes: convolutions, GEMMs, elementwise operations, fusion, and launch overhead. R50 became the preferred representative baseline to summarise the RT-DETR family.

---

## Conversation 4 — Latex Report Update

**Date:** 2026-06-05  
**Main theme:** Matching the professor’s acceleration-analysis requirements.

### Key discussion points

The professor’s requirement was quoted:

> Chacun doit avoir au minimum un modèle de sa famille accéléré avec TensorRT. Je veux que vous compariez le modèle avant et après accélération au niveau des sous-modules (convolution, batchnorm, etc.), et que vous montriez clairement d'où viennent les speed-ups.

You asked whether this information was already in the report.

Then the conversation explored TensorRT and Nsight terminology:

- What operations are executed in `enqueue` and in `myelinGraphExecute`?
- Comparison between `myelinGraphExecute` time and the PyTorch eager operations it fused.
- Whether PyTorch eager has an equivalent to TensorRT `enqueue`.
- Comparison between total PyTorch eager overhead and TensorRT enqueue time.
- What the `node_select` equivalent is in PyTorch eager kernels.
- Whether TensorRT GPU kernels are more directly comparable to PyTorch eager than TensorRT NVTX ranges.
- Difference between TensorRT GPU kernels and TensorRT ranges.

### Outcome / direction

The analysis distinguished between high-level TensorRT ranges such as `enqueue`/`myelinGraphExecute` and lower-level GPU kernels. It clarified that direct comparisons should be made carefully: TensorRT ranges may contain multiple fused operations, while PyTorch eager exposes many smaller ATen/cuDNN/GEMM kernels.

---

## Conversation 5 — RT-DETR Submodules

**Date:** 2026-06-12  
**Main theme:** RT-DETR architecture decomposition.

### Key discussion points

- Asked: “What are the submodules for RT-DETR?”

### Outcome / direction

The relevant RT-DETR decomposition used throughout the project became:

- Backbone: ResNet-style convolutional feature extractor.
- Encoder / hybrid encoder: multi-scale feature interaction and feature pyramid-style processing.
- Decoder: transformer object queries, cross-attention, FFN/GEMM-heavy blocks.
- Head: classification and box prediction layers.
- Post-processing: NMS-free RT-DETR decoding, score processing, box formatting.

---

## Conversation 6 — Nsight Profiler Script Issue

**Date:** 2026-06-09  
**Main theme:** Nsight profiling failures, kernel timing extraction, and INT8 feasibility.

### Key discussion points

- Shared a profiler command using ONNX Runtime TensorRT:

```bash
python -m benchmark.benchmark \
  --runtime ort_trt \
  --models rtdetr_ort_trt_r50 \
  --precision fp16 \
  --coco-images-dir /content/coco/val2017 \
  --ann-file /content/coco/annotations/instances_val2017.json \
  --subset-size 2000 \
  --subset-seed 42 \
  --batch-size 1 \
  --map-max-images ...
```

- Mentioned that the `nsys_stats` file did not work and asked how kernel-level timings were found.
- Shared an Nsight generation/import failure:

```text
Generating '/tmp/nsys-report-7835.qdstrm'
[1/8] [==21%] amp_fp16_compile_nsight_nsys_benchmark.nsys-rep
Importer error status: An unknown error occurred.
```

- Asked whether running INT8 ONNX TensorRT was already possible with the current codebase.
- Asked for an explanation of what INT8 is.

### Outcome / direction

The discussion focused on practical limitations of Nsight report generation in the environment, fallback ways to infer kernel timings, and the state of INT8 support. INT8 was framed as 8-bit quantised inference that can reduce memory bandwidth and accelerate supported convolution/linear operations, but requires calibration or quantisation-aware preparation and backend support.

---

## Conversation 7 — Final Report Summary

**Date:** 2026-06-09  
**Main theme:** Understanding professor expectations and analysing ORT TensorRT FP16 results.

### Key discussion points

- Asked for a summary of what the professor wants in the final report.
- Provided files/results for ORT TensorRT FP16 and requested help analysing the optimisation/acceleration part as expected by the professor.
- Asked for a side-by-side comparison of different kernel methods.
- Asked what `TensorRT:ExecutionContext::enqueue` and `myelinGraphExecute` mean, and whether they are convolution or attention.
- Requested a simple explanation.

### Outcome / direction

The expected final report was distilled into:

- Common protocol.
- Exact models and versions.
- Latency and mAP table.
- Submodule and operator-level profiling.
- GPU compute, memory transfer, kernel launch overhead, and CPU/Python analysis.
- Clear before/after acceleration comparison.
- Explanation of what TensorRT fused or accelerated.

TensorRT ranges were explained as execution containers rather than single operations: `enqueue` launches the TensorRT engine execution, while `myelinGraphExecute` corresponds to TensorRT/Myelin fused graph regions that may contain convolutions, GEMMs, elementwise operations, reshapes, and other fused blocks.

---

## Conversation 8 — AMP FP16 Optimisation Analysis

**Date:** 2026-06-05  
**Main theme:** Report writing based on torch.compile FP32 and AMP FP16 results.

### Key discussion points

- Corrected that the uploaded results were for `torch.compile(mode="reduce-overhead")`, FP32.
- Asked for analysis of how well the results satisfy the professor’s requirements.
- Repeated the professor’s requested additions for the next report version:
  - Missing mAP where possible.
  - Better introduction of the problem.
  - Clearer presentation of model families.
  - Dedicated section on elementary blocks: convolution, normalisation, attention, activations, NMS, RoIAlign, etc.
  - Mathematical definitions of important blocks.
  - Operator-level analysis, not just backbone/encoder/decoder.
  - First section on optimisation tools.
  - First acceleration experiments.
- Requested LaTeX code modifying the previous report rather than inventing a completely new structure.
- Asked to include a table of optimisations requested by the professor.
- Asked to mention that manual CUDA Graph capture was attempted but failed because of a CPU-to-CUDA copy inside the Hugging Face RT-DETR forward path.
- Asked what LaTeX document class should be used.

### Outcome / direction

The report structure was updated toward a more progressive narrative:

1. Problem statement and protocol.
2. Detector family overview.
3. Elementary operator definitions with mathematical formulas.
4. Baseline profiling by submodule and operator.
5. Optimisation tools considered.
6. First optimisation experiments.
7. Bottleneck interpretation and future work.

---

## Conversation 9 — Model Explanation for P6

**Date:** 2026-06-02  
**Main theme:** Explaining the current P6 model and profiling tools.

### Key discussion points

- Asked what the current model does.
- Asked whether `py-spy` is currently used.

### Outcome / direction

The current model was explained as an object detector in the RT-DETR / DETR-style family: it takes an image, extracts convolutional features, processes them through an encoder/decoder detection pipeline, and outputs object classes and bounding boxes. `py-spy` was clarified as a CPU/Python profiling tool, useful if Python-side overhead or post-processing is suspected, but not necessarily part of the current core profiling pipeline unless explicitly run.

---

## Cross-conversation technical decisions and recurring conclusions

### Common benchmark protocol

Across the chats, the benchmark protocol was standardised around:

- Dataset: COCO val2017.
- Subset: fixed 2000-image subset, seed 42.
- Batch size: 1.
- Warmup: 50 iterations.
- Timed iterations: 1000.
- Latency timer: CUDA events rather than plain `time.time()`.
- Hardware: NVIDIA Tesla T4.
- Profiling: `torch.profiler` plus Nsight Systems / Nsight Compute where possible.
- Output files: CSV summaries, traces, charts, and system breakdown files.

### RT-DETR model family

The main P6 focus became RT-DETR R18/R50/R101 from Hugging Face:

- `PekingU/rtdetr_r18vd`
- `PekingU/rtdetr_r50vd`
- `PekingU/rtdetr_r101vd`

The models share the same RT-DETR detection paradigm but differ in ResNet backbone depth.

### Baseline bottleneck interpretation

The project analysis repeatedly converged on:

- R18: encoder/decoder dominated; transformer-side optimisation matters most.
- R50: mixed bottleneck between backbone and decoder; good representative baseline.
- R101: backbone dominated; convolution-heavy acceleration strategies become more profitable.

### Operator-level conclusions

Important operator classes discussed repeatedly:

- `Conv2d` / cuDNN convolution: dominant in RT-DETR backbones.
- ReLU and residual additions: individually cheap but repeated often.
- GEMM / `addmm` / SGEMM: important in encoder/decoder linear projections and FFNs.
- LayerNorm and attention-related operations: more relevant in transformer blocks than in the convolutional backbone.
- Post-processing: present but RT-DETR is NMS-free, so there is no classic NMS bottleneck.
- Kernel launch overhead: important at batch size 1 and should be validated with Nsight Systems.

### Optimisation conclusions

The first optimisation experiments and analysis focused on:

- `torch.compile(mode="reduce-overhead")`: consistently useful for reducing eager overhead, especially at batch size 1.
- AMP FP16: model-size dependent; slower for small R18, neutral for R50, useful for heavier R101.
- CUDA Graphs: promising for launch overhead, but manual capture failed for the Hugging Face RT-DETR path because of an internal CPU-to-CUDA copy.
- TensorRT / ONNX Runtime TensorRT EP: required direction for stronger acceleration analysis and professor expectations.
- INT8: future or partial work depending on calibration/export/backend readiness.

### Report-writing direction

The report was progressively shaped to satisfy the professor’s feedback:

- Start with the problem and experimental protocol before discussing optimisations.
- Present detector families clearly.
- Include mathematical definitions for key elementary operators.
- Analyse at operator level, not only module level.
- Include optimisation tables with backend, precision, latency, FPS, mAP, speedup, and remarks.
- Explain why speedups happen: fusion, FP16/INT8, reduced launch overhead, TensorRT graph execution, or reduced memory movement.
- Be explicit about what remains unverified and what requires Nsight validation.

---

## Most important commands / command patterns discussed

### Quick RT-DETR benchmark

```bash
python -m benchmark.benchmark --models p6_hf --bench-iters 10 --warmup-iters 3 --skip-profiler --skip-viz --no-zip
```

### Full P6 benchmark

```bash
python -m benchmark.benchmark --models p6
```

### ORT TensorRT FP16 benchmark pattern

```bash
python -m benchmark.benchmark \
  --runtime ort_trt \
  --models rtdetr_ort_trt_r50 \
  --precision fp16 \
  --coco-images-dir /content/coco/val2017 \
  --ann-file /content/coco/annotations/instances_val2017.json \
  --subset-size 2000 \
  --subset-seed 42 \
  --batch-size 1
```

---

## Open issues / unfinished items from the chats

- Confirm whether the reorganised `benchmark` folder is redundant after moving runnable code under `utils/rt-detr`.
- Fully validate Nsight Systems reports when `.nsys-rep` import/export errors occur.
- Replace torch.profiler-based system estimates with proper Nsight Systems CUDA API attribution.
- Complete TensorRT before/after analysis with direct mapping from PyTorch eager kernels to TensorRT fused regions.
- Add at least one TensorRT-accelerated model in the final report, as required by the professor.
- Expand INT8 support only if calibration/export/runtime support is actually implemented and validated.
- Keep mAP measured on the same COCO subset for all baseline and optimised variants wherever possible.

