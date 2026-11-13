# AI Disclosure

AI tools were used in an assistive role during development. The core of the project (the Blender scene, the eye rig, the calibration geometry, the ground-truth computation and the verification) is deterministic code and Blender. Every AI-assisted change was reviewed, run and tested by the team.

## AI tools used

| Tool | Role |
|---|---|
| **OpenCode** with **GLM** (open-weight model, self-hosted on our college's infrastructure), plus a cloud coding assistant | Coding assistance for the wrapper and tooling; documentation and presentation edits, the architecture diagram and repository organisation |
| **Blender MCP** | Let the assistant inspect and tweak the scene, rig and materials in a running Blender session |
| **brag-slim** (AI video-generation skill for coding agents) | Produced the demo video: wrote the storyboard and the scripts that composite our own renders and synthesise the soundtrack |

## What is not AI-generated

- **Base head:** Google GNM Head v3 (Apache-2.0), a 3D head model converted to a metric world frame by `tools/export_gnm.py`.
- **Scene and rig:** eye bones, materials, the locked camera and the static lights are built in code by `tools/build_scene.py`.
- **Textures:** all 8 sets are generated procedurally by `tools/make_textures.py` (NumPy, SciPy, OpenCV). No image-generation model was used.
- **Gaze geometry:** each eye's rotation is computed analytically from the fixation target `T(h,v)` in `wrapper/gaze.py`.
- **Eye motion:** saccades, blinks and the eye roll are procedural curves with a fixed random seed in `wrapper/motion.py`, not a learned motion model.
- **Calibration guard:** `wrapper/eyes.py` checks the camera position, focal length and render size before every render and refuses to render if any of them change.
- **Renders and videos:** path-traced in Blender Cycles from `scene.blend` through `wrapper/eyes.py`. Cycles' built-in OpenImageDenoise filter only removes sampling noise; it does not generate content.
- **Ground truth:** `metadata.json` is computed analytically (fixation geometry and the projection `P = K[R|t]`), not predicted by a model.
- **Direction check:** `tools/qa_pupil_check.py` uses classical image processing (blur, thresholding, connected components), not a neural network.
- **Demo video footage:** all footage comes from our own renders and videos, every number on screen was re-measured by our tools before rendering, and the soundtrack is synthesised procedurally in code. No generative video or audio model was used.

## How we verified AI-assisted work

- `tests/test_math_in_blender.py`: clockwise table, gaze geometry and camera projection
- `tools/qa_pupil_check.py`: independent pupil detection, **5 / 5 directions pass**, mean error **4.48 px**
- `tools/validate_submission.py`: **25 / 25** bundle checks pass
