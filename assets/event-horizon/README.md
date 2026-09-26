# Event Horizon — banner renderer

The profile banner is not an illustration. It's a render of a Schwarzschild black hole, made by tracing
12.8 million light rays (16 per pixel) through curved spacetime on the GPU.

- **Geodesics**: each ray follows the exact photon-orbit equation `d²u/dφ² + u = 3Mu²`, integrated with RK4
  in PyTorch ([`trace.py`](trace.py)). What comes out is real lensing: the far side of the disk bends over and under the
  shadow, a photon ring forms at `r = 3M`, and the star field and Milky Way are warped around the hole.
- **Accretion disk**: a thin disk from the ISCO (`6M`) out to `26M`. It glows as a blackbody, with a Novikov–Thorne-style
  temperature profile plus Doppler and gravitational redshift, so the approaching side burns whiter
  ([`shade.py`](shade.py)).
- **Motion**: the disk turns with Keplerian speed (`Ω ∝ r^-3/2`). Two flow-map texture layers crossfade, which lets the
  8-second animation loop with no seam.
- **Finish**: bloom, an anamorphic streak and ACES tone-mapping, then the typography layer ([`typeset.py`](typeset.py)).

```bash
python assets/event-horizon/build.py          # -> assets/banner-event-horizon.avif (+ .webp fallback)
python assets/event-horizon/build.py --still  # -> assets/banner-event-horizon-still.png
```

To change the text, edit `SPEC` in [`build.py`](build.py). Requirements: a CUDA GPU, `torch`, `numpy`, `pillow`, and
`ffmpeg` built with `libaom-av1`.
