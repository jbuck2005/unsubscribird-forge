# Bird recognition on the NPU (plan)

Status: **planning**, nothing built yet (2026-11-24). It builds on the open NPU stack of
[jbuck2005/orangepi6](https://github.com/jbuck2005/orangepi6) ([orangepi6 docs/npu.md](https://github.com/jbuck2005/orangepi6/blob/main/docs/npu.md)).

## Goal

Run a bird feeder station's two heavy model calls on the Orange Pi 6's NPU:

1. **Detection**: find the bird in a frame (today: OWL-ViT base patch32 on the CPU, about 1 s a frame).
2. **Embedding**: turn each frame into BioCLIP 2 image embeddings (a ViT-L/14 CLIP, 4 or 5 square views a frame:
   the whole frame letterboxed, left, centre, right, and a crop around the detected bird). A small classifier
   trained on those embeddings names the species.

The model work (export, compiling, packs, training, evaluation) lives in this repo, and the board's kernel, NPU
driver and library come from orangepi6. The station software (un-subscribird) gets no NPU code; how it uses what this
repo provides is an open question below.

## What we already have

From [orangepi6 docs/npu.md](https://github.com/jbuck2005/orangepi6/blob/main/docs/npu.md), working since 2026-11-22 on both kernel tracks:

- Kernel driver CIX 6.2.0 (GPL-2.0) and Arm China `libaipudrv` 4.1.0 (Apache-2.0). No firmware, no vendor kernel,
  no closed runtime.
- orangepi6's [`tools/npu-run.cpp`](https://github.com/jbuck2005/orangepi6/blob/main/tools/npu-run.cpp) loads a compiled `.cix` graph and runs it; `bench` and `classify` work. MobileNetV2 runs at
  about 1.6-1.8 ms.
- Compiled graphs come from CIX's model hub (`cix/ai_model_hub_26_Q2` on ModelScope). The compiler (CixBuilder) is
  closed and x86-64 only, needed only for our own models.
- The closed `libnoe` and ONNX Runtime provider are not used, and this plan doesn't need them.

## What is known about these models on this NPU

From other people's results on the same SoC (different boards, CIX's closed stack), so treat as a guide only:

| Model | Relevance | Time |
|---|---|---|
| CLIP ViT-B/32 (image and text graphs on the hub) | stand-in for BioCLIP 2 | 13-17 ms an image [1][2] |
| SigLIP so400m-384 | upper bound for a big ViT | about 1.1 s an image [1] |
| YOLO-World (open vocabulary) | OWL-ViT replacement | about 1 s (two passes) [1] |
| YOLOv8-L | small closed-set detector | about 44 ms [1][3] |

- After INT8, CLIP image embeddings had cosine 0.85 to 0.97 against FP32, and an INT8 CLIP text tower was badly
  wrong, so text embeddings should stay FP32 on the CPU [1].
- Nobody has shown OWL-ViT or BioCLIP 2 on this NPU.
- **Estimate, not measured:** BioCLIP 2 (ViT-L/14 at 224 px) is roughly 4 to 5 times the compute of ViT-B/32, so
  perhaps 0.1 to 0.4 s a view, or 0.5 to 2 s a frame for 5 views. A fixed batch of 5 has to be compiled in.

## Design

### Model packs

A pack is a folder: the compiled `.cix` graph plus a `manifest.json` with the source model and revision, compiler
version, input shape and preprocessing (size, mean, std, quantisation scale and zero point), output (embedding size,
or box format and labels), precision, and the SHA-256 of each file. The loader refuses a pack whose checksum or
manifest version it doesn't know. Packs live in a `models/` folder; the graphs themselves are not committed (they can
be large, and see Privacy).

### Running a pack

`npu-run classify` handles one input and a top-5 output. Embeddings and boxes need more:

- read all outputs and dequantise them to float32 with each output's scale and zero point,
- feed a batch of views in one job,
- keep the graph loaded between frames (load and job creation cost 4 ms here).

Options: extend `npu-run` with an `embed` command and a long-running mode, or add a small Python module over
`libaipudrv` with `ctypes`. The second suits an image pipeline better. Either way it stays on the open stack.

### Detection

1. OWL-ViT exported to ONNX with its "bird" / "squirrel" queries baked in, then compiled. Keeps today's behaviour;
   its detection heads may not compile well.
2. A small detector of our own (YOLO-class, 640 px) trained on boxes a station already has (its OWL-ViT detections
   and hand-picked training frames). Fastest and most robust, and it can learn the small birds OWL-ViT misses. The
   hub has no YOLO graph, so we compile it ourselves, and the detector family's licence has to be checked (some YOLO
   code is AGPL).

Box post-processing stays on the CPU in numpy, whatever the model.

### Compiling

CixBuilder runs on an x86-64 Linux machine (Python 3.10). Plan: a pinned container for it, run on my x86 desktop in
WSL2. Input is a fixed-shape ONNX file plus a `.cfg` (INT8 calibration set, per-layer FP16/INT16 overrides, target
`X2_1204MP3`) [3][4]. Output is the `.cix` graph for a pack.

## Phases

| Phase | What | Done when |
|---|---|---|
| 1. CLIP from the hub | Run the hub's CLIP image graph with dequantised float outputs; compare its embeddings with FP32 CLIP on the CPU for the same images. | Cosine figures and timings for a known model |
| 2. Pack runner | `embed` support (multi-output, batch, long-running), as a Python module or `npu-run` mode; manifest checks. | A pack loads and embeds a folder of images |
| 3. BioCLIP 2 | Export the image tower to ONNX, compile INT8 and FP16 (or mixed) on the x86 desktop, compare with FP32 on the CPU. | Speed and accuracy table per precision |
| 4. Detector | OWL-ViT compiled, and a small detector of our own trained and compiled; compare on the same frames. | A detector fast on the NPU and at least as good on small birds |
| 5. Accuracy check | Retrain the species classifier on NPU embeddings and compare visit accuracy with the CPU model on the same visits. | No more than 1 to 2 % lower |
| 6. Integration | However the station uses it (see open questions), plus packaging like orangepi6's `.deb` packages. | Running beside the current station for a week |

Phase 1 needs nothing but the board and the hub.

## Risks

- **INT8 accuracy.** Training doesn't cost accuracy; compiling to INT8 can. CNN detectors usually lose under a
  point or two of mAP with a good calibration set. ViT embedders are touchier; FP16 or mixed precision for the image
  tower is the fallback, and retraining the classifier on NPU embeddings absorbs most of the drift.
- **Transformer ops.** ViTs compile on this NPU (CLIP, SigLIP), but OWL-ViT's heads may not.
- **Closed compiler.** CixBuilder is closed and x86-only, and newer releases have made models slower for some
  users [3]. Pin the version that works.
- **Library ABI.** `libaipudrv` 4.2.0 and later changed the ABI, so we stay on 4.1.0 with the 6.2.0 driver (see
  [orangepi6 docs/npu.md](https://github.com/jbuck2005/orangepi6/blob/main/docs/npu.md)).
- **Redistribution.** The licence of compiled graphs from CixBuilder, and of each source model's derived weights,
  needs checking before any pack is published.

## Privacy

Calibration and training frames come from real households. They stay on the machine that compiles or trains, and
are never committed. Packs built only from public models (BioCLIP 2, OWL-ViT) may be published once their licences
allow it. Packs trained or calibrated on one household's frames stay with that household: model weights can carry
traces of the images they were trained on.

## Decisions so far

- Board: Orange Pi 6, 16 GB, beside the current station host, not replacing it.
- Up to 1 to 2 % lower visit accuracy than the CPU model is acceptable.
- Train our own small detector.
- Compile on my x86 desktop.
- Publish packs from public models only once licences allow; self-trained packs stay private.
- No NPU or CPU backend code in un-subscribird.

## Open questions

1. How should the station use the NPU? A small local inference service on the board (HTTP or Unix socket) that the
   station calls, a standalone tool that processes the station's frames, or something else?
2. Where does the pack runner (phase 2) live: here as a Python module, or in orangepi6 as an `npu-run` mode?

## Sources

1. CIX P1 NPU model zoo (third party, MS-R1 board): https://github.com/opivankristovi/MS-R1-NPU_model_zoo
2. Reference results from the Orange Pi 6 Plus on CIX's `libnoe` stack, as summarised in [orangepi6 docs/npu.md](https://github.com/jbuck2005/orangepi6/blob/main/docs/npu.md).
3. Radxa forum, CIXBuilder slow inference:
   https://forum.radxa.com/t/cixbuilder-problems-compiling-onnx-model-slow-inference-times/26972
4. Radxa, NPU environment setup and YOLOv8 example (same SoC):
   https://docs.radxa.com/en/orion/o6n/app-development/artificial-intelligence/env-setup and
   https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/yolov8
