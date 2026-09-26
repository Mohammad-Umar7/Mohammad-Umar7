"""
Builds assets/banner-event-horizon.{avif,webp}.

    python assets/event-horizon/build.py            # full render + encode (~3 min on an RTX 4070)
    python assets/event-horizon/build.py --still    # one frame -> banner-event-horizon-still.png

Needs: CUDA GPU, torch, numpy, pillow, and ffmpeg built with libaom-av1 on PATH.
Fonts (Syncopate Bold, JetBrains Mono) are fetched from google/fonts on first run.
"""
import os, sys, time, argparse, shutil, subprocess, tempfile, urllib.request
import numpy as np, torch
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from trace import trace
from shade import Disk, Sky, shade, sky_cache, gauss_blur, aces
from typeset import build_overlay, FD

DEV = "cuda"

# ------------------------------------------------------------------ scene
CFG = dict(W=1600, H=500, ss=4,                 # output size, ss x ss rays per pixel
           D=50.0, elev=5.0, azim=-90.0, roll=0.0,  # camera: distance (M), height above disk plane (deg)
           fov=72, cx=0.70, cy=0.53,            # hole sits at (cx, cy) of the frame
           r_in=6.0, r_out=26)                  # disk from the ISCO out to 26 M

P = dict(loop=8.0, spin=16.0, shear=40.0,       # 8 s seamless loop, flow-map rotation
         turb=1.6, ringamp=0.3, contrast=3.0, bias=0.2, tex_ax=14.0, tex_beta=1.0,
         hot_amt=1.5, hot_thr=1.0, disk_seed=7, disk_seedB=19,
         r_in_eff=5.4, T0=4700, Texp=0.45, t_lo=0.6,  # Novikov-Thorne-like temperature profile
         gT=0.8, gI=2.2,                         # tempered Doppler + gravitational redshift
         Ipow=1.0, gain=1.0, base=0.1, alpha=0.92, a_inner=0.6, fade_w=6.0, ragged=1.0,
         emis_scale=1.0, thin_glow=0.15,
         exposure=1.5, bloom=[(2, 0.06), (8, 0.04), (24, 0.02), (60, 0.01)],
         streak_thr=2.5, streak_sigma=260, streak_tint=(0.9, 0.75, 0.6), streak_amt=0.25,
         vignette=0.35, sat=1.2, hue_preserve=0.5)

SKY = dict(star_layers=[(14, 0.5, 0.008, 1.1, 0.6), (34, 0.55, 0.025, 1.3, 0.65), (90, 0.7, 0.08, 1.6, 0.75)],
           mw_normal=(-0.423, 0.0, -0.906), mw_gain=0.045, mw_width=0.16, neb_gain=0.004, neb2_gain=0.0)

# ------------------------------------------------------------------ typography
MONO = "JetBrainsMono[wght].ttf"
TITLE = "Syncopate-Bold.ttf"
AMBER, MUTED, DIM = (255, 166, 77), (178, 188, 204), (120, 130, 150)
X = 96
SPEC = [
    dict(text="CHIEF AI OFFICER  @  KANBAN STUDIOS", font=MONO, var=500, size=20, track=4.0, x=X + 2, y=118, color=AMBER),
    dict(text="MOHAMMAD", font=TITLE, size=66, track=7, x=X, y=196, grad=((255, 255, 255), (255, 228, 196)), glow=0.9),
    dict(text="UMAR", font=TITLE, size=66, track=7, x=X, y=270, grad=((255, 255, 255), (255, 228, 196)), glow=0.9),
    dict(text="AGENTIC AI  ·  RAG  ·  VOICE AI  ·  FULL-STACK", font=MONO, var=400, size=19, track=2.0, x=X + 2, y=338, color=MUTED),
    dict(text="d²u/dφ² + u = 3Mu²   ·   every photon traced through curved spacetime", font=MONO, var=300, size=15,
         track=0.8, x=X + 2, y=452, color=DIM, alpha=0.95),
]
OVERLAY = dict(glow_color=(255, 170, 90), glow_radius=14, glow_amt=0.5, shadow_amt=0.55, shadow_radius=10, scrim=(0.35, 40))

FONT_URLS = {
    TITLE: "https://raw.githubusercontent.com/google/fonts/main/apache/syncopate/Syncopate-Bold.ttf",
    MONO: "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
}


def ensure_fonts():
    os.makedirs(FD, exist_ok=True)
    for name, url in FONT_URLS.items():
        p = os.path.join(FD, name)
        if not os.path.exists(p):
            print("fetching", name)
            urllib.request.urlretrieve(url, p)


# ------------------------------------------------------------------ post
def glow_static(mean_hdr, P):
    """large-scale bloom + anamorphic streak from the time-averaged frame, so the
    sky pixels are bit-identical across frames (keeps the animation small)."""
    W = mean_hdr.shape[1]
    x = mean_hdr * P["exposure"]
    glow = torch.zeros_like(x)
    for s, w in P["bloom"]:
        if s > 3:
            glow += w * gauss_blur(x, s * W / 1600)
    br = (x + glow - P["streak_thr"]).clamp(min=0)
    streak = gauss_blur(gauss_blur(br, P["streak_sigma"] * W / 1600, axis="x"), 1.2 * W / 1600, axis="y")
    return glow + streak * torch.tensor(P["streak_tint"], device=x.device) * P["streak_amt"]


def tonemap(hdr, glow, P):
    H, W = hdr.shape[:2]
    x = hdr * P["exposure"]
    for s, w in P["bloom"]:
        if s <= 3:
            glow = glow + w * gauss_blur(x, s * W / 1600)
    x = x + glow
    yy = torch.linspace(-1, 1, H, device=x.device)[:, None]; xx = torch.linspace(-1, 1, W, device=x.device)[None, :]
    x = x * (1 - P["vignette"] * ((xx * 0.85) ** 2 + (yy * 0.6) ** 2).clamp(0, 1))[..., None]
    lw = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device)
    lum = (x * lw).sum(-1, keepdim=True)
    x = (lum + (x - lum) * P["sat"]).clamp(min=0)
    y1 = aces(x).clamp(0, 1)
    L = (x * lw).sum(-1, keepdim=True).clamp(min=1e-6)            # hue-preserving variant
    y2 = x * (aces(L) / L)
    mx = y2.max(-1, keepdim=True).values
    y2 = torch.where(mx > 1, y2 / mx + (1 - 1 / mx) * 0.9, y2).clamp(0, 1)
    y = y1 * (1 - P["hue_preserve"]) + y2 * P["hue_preserve"]
    y = torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
    return y.clamp(0, 1)


class Overlay:
    def __init__(self, W, H, spec, glow_color, glow_radius, glow_amt, shadow_amt, shadow_radius, scrim):
        txt, gm = build_overlay(W, H, spec)
        blur = lambda a, r: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(r))) / 255.0
        self.a = torch.tensor(txt[..., 3:4] / 255.0, dtype=torch.float32, device=DEV)
        self.col = torch.tensor(txt[..., :3] / 255.0, dtype=torch.float32, device=DEV)
        dark = shadow_amt * blur(txt[..., 3], shadow_radius)
        sc = blur(txt[..., 3], scrim[1]); sc = np.clip(sc / max(sc.max(), 1e-6), 0, 1)
        dark = 1 - (1 - dark) * (1 - scrim[0] * sc)
        self.dark = torch.tensor(dark[..., None], dtype=torch.float32, device=DEV)
        gl = blur(gm, glow_radius) * glow_amt
        self.glow = torch.tensor(gl[..., None], dtype=torch.float32, device=DEV) * torch.tensor(glow_color, device=DEV) / 255.0

    def apply(self, y):
        y = y * (1 - self.dark)
        y = 1 - (1 - y) * (1 - self.glow)
        return y * (1 - self.a) + self.col * self.a


# ------------------------------------------------------------------ pipeline
def render_frames(nframes, frame_dir=None, only=None):
    t0 = time.time()
    buf = trace(CFG)
    print(f"traced {CFG['W']*CFG['H']*CFG['ss']**2/1e6:.1f}M geodesics in {time.time()-t0:.0f}s")
    disk = Disk(CFG["r_in"], CFG["r_out"], seed=P["disk_seed"], seedB=P["disk_seedB"], ax=P["tex_ax"], beta=P["tex_beta"])
    sky = Sky(CFG["fov"], CFG["W"], **SKY)
    sc = sky_cache(buf, sky)
    idx = list(range(0, nframes, max(1, nframes // 24)))
    mean = sum(shade(buf, disk, sky, P, P["loop"] * i / nframes, skycache=sc) for i in idx) / len(idx)
    glow = glow_static(mean, P)
    ov = Overlay(CFG["W"], CFG["H"], SPEC, **OVERLAY)
    out = []
    for i in (range(nframes) if only is None else only):
        y = ov.apply(tonemap(shade(buf, disk, sky, P, P["loop"] * i / nframes, skycache=sc), glow, P))
        im = Image.fromarray((y * 255 + 0.5).clamp(0, 255).byte().cpu().numpy())
        if frame_dir:
            im.save(os.path.join(frame_dir, f"f{i:04d}.png"), compress_level=1)
        out.append(im)
    print(f"shaded {len(out)} frames in {time.time()-t0:.0f}s total")
    return out


def encode_avif(frame_dir, out, fps, crf=27):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(frame_dir, "f%04d.png"), "-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0",
                    "-cpu-used", "3", "-row-mt", "1", "-pix_fmt", "yuv420p10le", "-color_primaries", "bt709",
                    "-color_trc", "iec61966-2-1", "-colorspace", "bt709", "-color_range", "pc", "-loop", "0",
                    "-f", "avif", out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true", help="render a single PNG frame instead")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    ensure_fonts()
    nframes = int(P["loop"] * a.fps)
    if a.still:
        render_frames(nframes, only=[nframes // 2])[0].save(os.path.join(ASSETS, "banner-event-horizon-still.png"))
        return
    tmp = tempfile.mkdtemp(prefix="event-horizon-")
    try:
        frames = render_frames(nframes, frame_dir=tmp)
        avif = os.path.join(ASSETS, "banner-event-horizon.avif")
        encode_avif(tmp, avif, a.fps)
        print(f"avif  {os.path.getsize(avif)/1e6:.2f} MB  ({nframes} frames @ {a.fps} fps)")
        # fallback for browsers without AVIF: 1280 px, half frame rate
        webp = os.path.join(ASSETS, "banner-event-horizon.webp")
        small = [f.resize((1280, 400), Image.LANCZOS) for f in frames[::2]]
        small[0].save(webp, save_all=True, append_images=small[1:], duration=round(2000 / a.fps), loop=0,
                      quality=70, method=6, minimize_size=True)
        print(f"webp  {os.path.getsize(webp)/1e6:.2f} MB  ({len(small)} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
