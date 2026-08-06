# MiniMax H3 FirstBlockCache — benchmark

## Fixed setup

- Backend: SageAttention 2.2.0 (`--use-sage-attention`)
- Workflow: native MiniMax H3 T2V
- Output: 960×544, 0.5 MP, 5 seconds (124 frames), 24 fps
- Sampling: 20 steps, `res_multistep`, fixed seed
- Checkpoint: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Prompt and seed: identical for every row (`20260807`)
- Cold: `/free` unload before the run; warm: immediate same-shape repeat
- Headline speedup: warm cache run divided only by warm no-cache run on the same attention backend

## Runs

### SageAttention2 matrix

Raw wall times:

| Configuration | Cold | Warm | Cache hits |
|---|---:|---:|---:|
| No cache | 70.517 s | 58.389 s | — |
| Safe | 56.245 s | 46.371 s | 6/20 |
| Fast | 52.372 s | 40.261 s | 8/20 |
| Aggressive | 47.294 s | 37.277 s | 10/20 |
| No cache confirmation | — | 57.527 s | — |

The two warm no-cache measurements bracketed the preset matrix (`58.389` and `57.527` seconds), so their mean `57.958 s` is used for the robust cache-only summary:

| Preset | Warm time | Cache-only speedup | Wall-time reduction |
|---|---:|---:|---:|
| Safe | 46.371 s | 1.25× | 20.0% |
| Fast | 40.261 s | 1.44× | 30.5% |
| Aggressive | 37.277 s | 1.55× | 35.7% |

### Native attention control

| Configuration | Cold | Warm | Warm cache-only speedup |
|---|---:|---:|---:|
| No cache | 100.808 s | 90.644 s | — |
| Fast | 70.476 s | 60.819 s | **1.49× / 32.9% less time** |

The similar Fast gain on native (`1.49×`) and SageAttention2 (`1.44×`) shows that the cache result is not a SageAttention artifact. SageAttention acceleration itself is not included in the cache-only number.

## Output review

[Warm comparison grid](results/sage2_warm_comparison_grid.jpg)

https://github.com/user-attachments/assets/ad504313-6a94-44ab-b3c5-beca2511e4cd

All four outputs are coherent and free of black tiles, checkerboards, or collapse. Fast keeps the same scene and action while changing the exact generative trajectory. Aggressive changes framing and motion more visibly, so it remains an opt-in speed preset rather than the default.

Decoded-frame comparison against the warm no-cache output:

| Preset | SSIM | PSNR |
|---|---:|---:|
| Safe | 0.8216 | 25.80 dB |
| Fast | 0.6873 | 22.09 dB |
| Aggressive | 0.6159 | 19.48 dB |

These metrics measure pixel alignment, not perceptual video quality; small motion shifts lower them sharply. They are recorded as trajectory-distance evidence, not as a claim that Fast has poor visual quality. For every configuration, cold and warm decoded videos matched exactly (`SSIM 1.000`), confirming deterministic A/B execution.

The post-change `Custom` smoke test used 608×352 / 39 frames on native attention. With threshold `0.10`, window `0.10–0.95`, max chain `2`, and temporal guard enabled, it completed in `25.112 s`, cached `6/20` steps, reported `temporal guard max 0.24316`, produced 39 valid frames, and had no execution errors or black-frame detections. This is a functional guard test, not a speed row comparable to the 0.5 MP matrix.

## Conclusion

Keep three presets. `H3 Fast` is the default: it gave about 30–33% lower warm wall time on both tested attention backends without structural failure in the visual review. `Safe` is the conservative option; `Aggressive` is the maximum-speed option. A separate `Custom` mode exposes the decision threshold, protected window, hit-chain limit, and an optional target-video temporal guard without changing the calibrated presets.
