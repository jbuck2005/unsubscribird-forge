# History

## 2026-11-24: repository started

- `README.md`: what this repository is for.

## 2026-11-24: bird recognition plan

- `docs/bird-recognition.md`: plan for running the station's detection (OWL-ViT today) and BioCLIP 2 species
  embeddings on the Orange Pi 6's NPU, on orangepi6's open stack. Phases start with the hub's CLIP graph, then a pack
  runner with float outputs, BioCLIP 2 and a detector compiled with CixBuilder, and an accuracy check (no more than
  1 to 2 % below the CPU model). Planning only; nothing built. Moved here from orangepi6, where it was first drafted.

## Open items

- Bird recognition: open questions in `docs/bird-recognition.md` (how the station uses the NPU; where the pack
  runner lives).
