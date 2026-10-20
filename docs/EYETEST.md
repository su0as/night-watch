# Gemma eyesight test

Generated 2026-10-21 00:35 by `python -m nightwatch eyetest` (robot `HalfCheetah-v5`).

The robot is dropped into a known pose (the right way up, or on its back), the physics settles it with small random motor twitches, and four snapshots are taken, exactly like a test episode during training. The truth is the torso angle from the simulator (upright < 60 degrees, upside down > 120). The judge is the same code that watches at night; only what it is shown changes.

| model | camera | frame size | floor | right wake decision | upright left alone | upside down caught | image+prompt tokens per test episode | seconds per test episode |
|---|---|---|---|---|---|---|---|---|
| `gemma4:e4b-it-qat` | head camera, 4 snapshots | 512x384 | matte | **18 / 18** | 12 / 12 | 6 / 6 | 796 | 25.1 |
| `gemma4:e4b-it-qat` | side camera, 2 snapshot(s) | 1024x768 | matte | **12 / 18** | 12 / 12 | 0 / 6 | 960 | 22.3 |
| `gemma4:e4b-it-qat` | side sheet | 320x240 | matte | **6 / 12** | 6 / 6 | 0 / 6 | 423 | 10.3 |

What Gemma called each pose:

- 1024x768, head camera, 4 snapshots, head (`head-v3-n4`): truly upright: upright x11, unclear x1; truly upside_down: upside_down x6
- 1024x768, side camera, 2 snapshot(s), side (`v3-n2-px1024`): truly upright: upright x12; truly upside_down: upright x6
- 320x240, side sheet, side (`v3-n0-px1024`): truly upright: upright x6; truly upside_down: upright x6

Ollama gives Gemma 4 roughly one image token per 48x48 pixels, so a 2x2 sheet of 320x240 snapshots is ~130 image tokens for all four snapshots, while one 1024x768 snapshot is ~336. Seconds per look were measured on an Intel i7-1165G7 laptop with Gemma on the Iris Xe iGPU (Vulkan).
