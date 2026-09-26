# Event Horizon — banner renderer

The profile banner is not an illustration. It's a render of a Schwarzschild black hole: 12.8 million light rays
(16 per pixel) traced through curved spacetime for every one of its 360 frames.

- **Lensing** ([`lens.py`](lens.py)): each photon moves in a plane through the hole, and in that plane its path depends
  only on its impact parameter `b`. So the photon-orbit equation `d²u/dφ² + u = 3Mu²` is integrated once, in float64,
  for 12k values of `b`. After that, each pixel only needs its orbital plane and the angles where that plane cuts the
  disk (the line of nodes), which is a few table lookups. That turns seconds of ray marching into milliseconds per frame.
  This is what lets the camera move.
- **Camera**: it swoops through the disk plane, from 7.5° above to 6° below. The lensed image flips as it goes, and
  when the camera is level the far side of the disk closes into a full ring. At the same time the camera orbits the hole,
  so the lensed stars stream past and whip around it.
- **Accretion disk** ([`shade.py`](shade.py)): a thin disk from the ISCO (`6M`) out to `24M`. It glows as a blackbody
  with Doppler and gravitational redshift, so the approaching side burns whiter. It turns at Keplerian speed
  (`Ω ∝ r^-3/2`) through crossfading flow-map layers. It also carries a rotating two-armed spiral wave and hot-spot
  flares that complete whole orbits per loop, so their lensed light races over the top of the hole.
- **Motion blur**: the 16 samples in each pixel are spread across the shutter interval.
- **Seamless loop**: every motion is periodic in 12 s — the camera path, the orbits, the flow cycles, the spiral wave,
  and the sky, which has 6-fold symmetry so a 60° turn lines up exactly.
- **Finish**: dynamic bloom, a soft anamorphic streak, ACES tone-mapping, then animated type ([`typeset.py`](typeset.py)):
  a light sweep across the name, tags lighting up in turn, and a blinking cursor.

```bash
python assets/event-horizon/build.py            # -> assets/banner-event-horizon.avif (+ .webp fallback)
python assets/event-horizon/build.py --still 3  # -> assets/banner-event-horizon-still.png (frame at t = 3 s)
```

To change the text, edit `OVERLINE`, `NAME`, `TAGS` or `CAPTION` in [`build.py`](build.py). Requirements: a CUDA GPU,
`torch`, `numpy`, `pillow`, and `ffmpeg` built with `libaom-av1`.
