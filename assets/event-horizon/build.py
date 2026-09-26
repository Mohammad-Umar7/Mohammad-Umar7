"""
Builds assets/banner-event-horizon.avif (+ .webp fallback).

    python assets/event-horizon/build.py            # full render + encode (~10 min on an RTX 4070)
    python assets/event-horizon/build.py --still 3  # one frame at t = 3 s -> banner-event-horizon-still.png

The camera swoops through the disk plane (elevation -6 deg .. +7.5 deg) while orbiting the hole
(the 6-fold symmetric sky turns 60 deg per loop), the disk swirls at Keplerian speed with orbiting
flares, and every sample carries its own shutter time for motion blur. 12 s seamless loop, 30 fps.

Needs: CUDA GPU, torch, numpy, pillow, and ffmpeg built with libaom-av1 on PATH.
Fonts (Syncopate Bold, JetBrains Mono) are fetched from google/fonts on first run.
"""
import os, sys, math, time, argparse, shutil, subprocess, tempfile, urllib.request
import numpy as np, torch
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from lens import OrbitTable, lens_rays
from shade import Disk, SkyP, disk_emission, tonemap, bayer, TAU
from typeset import build_overlay, font, text_width, FD

DEV = "cuda"

# ------------------------------------------------------------------ scene
CFG = dict(W=1600, H=500, ss=4,                  # output size, ss x ss rays per pixel
           D=50.0, azim=-90.0, roll=0.0,         # camera distance (M); elevation is animated
           fov=74, cx=0.72, cy=0.53,             # hole sits at (cx, cy) of the frame
           r_in=6.0, r_out=24)                   # disk from the ISCO out to 24 M
FPS = 30

P = dict(loop=12.0,                               # seconds; every motion below repeats exactly
         elev_mid=0.75, elev_amp=6.75,            # camera elevation swing through the disk plane (deg)
         flow_cycles=4, rate=19.39, shear=40.0,   # Keplerian flow-map; rate puts r=7 at 2 orbits/loop
         turb=1.6, ringamp=0.3, contrast=3.0, bias=0.2, hot_amt=1.5, hot_thr=1.0, tex_ax=4.0, tex_beta=1.25,
         arm_amp=0.6, arm_m=2, arm_wind=4.0, arm_sharp=2.0, arm_turns=2,   # rotating two-armed spiral wave
         r_in_eff=5.4, T0=4700, Texp=0.45, t_lo=0.6, gT=0.8, gI=2.2, Ipow=1.0, gain=1.0, base=0.1,
         alpha=0.92, a_lo=0.55, a_inner=0.6, fade_w=6.0, ragged=1.0, emis_scale=1.0, thin_glow=0.15,
         flare_heat=1.6,
         # (radius, orbits per loop, phase, amplitude, radial sigma, lead sigma, tail length, tail drift)
         flares=[(11.11, 1, 0.0, 14.0, 0.9, 0.14, 1.1, 0.6), (11.11, 1, 2.1, 9.0, 0.7, 0.12, 0.8, 0.5),
                 (7.0, 2, 4.2, 8.0, 0.5, 0.10, 0.7, 0.35)],
         exposure=1.25, bloom=[(2, 0.06), (8, 0.04), (24, 0.02), (60, 0.01)],
         streak_thr=2.5, streak_sigma=260, streak_tint=(0.9, 0.75, 0.6), streak_amt=0.18, streak_cap=4.0, streak_sy=3.5,
         vignette=0.35, sat=1.2, hue_preserve=0.5, sky_dir=1.0)

SKY = dict(fold=6, star_layers=[(14, 0.5, 0.008, 1.1, 0.6), (34, 0.55, 0.025, 1.3, 0.65), (90, 0.7, 0.07, 1.6, 0.75)],
           neb_gain=0.004, neb2_gain=0.0, band_gain=0.02, band_z=0.05, band_w=0.16)

# ------------------------------------------------------------------ typography
MONO = "JetBrainsMono[wght].ttf"
TITLE = "Syncopate-Bold.ttf"
AMBER, MUTED, DIM = (255, 166, 77), (178, 188, 204), (120, 130, 150)
X = 96
OVERLINE = "CHIEF AI OFFICER  @  KANBAN STUDIOS"
NAME = ("MOHAMMAD", "UMAR")
TAGS = ["AGENTIC AI", "RAG", "VOICE AI", "FULL-STACK"]
CAPTION = "d²u/dφ² + u = 3Mu²   ·   every photon traced through curved spacetime"

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


def elevation(t):
    return P["elev_mid"] + P["elev_amp"] * math.sin(TAU * t / P["loop"])


# ------------------------------------------------------------------ renderer
class Renderer:
    def __init__(self):
        W, H, ss = CFG["W"], CFG["H"], CFG["ss"]
        f = (W / 2) / math.tan(math.radians(CFG["fov"]) / 2)
        far = math.hypot(max(CFG["cx"], 1 - CFG["cx"]) * W, max(CFG["cy"], 1 - CFG["cy"]) * H)
        t0 = time.time()
        self.tab = OrbitTable(CFG["D"], CFG["D"] * math.sin(math.atan(far / f)) + 2.0)
        print(f"orbit table {tuple(self.tab.U.shape)} in {time.time()-t0:.1f}s")
        self.disk = Disk(CFG["r_in"], CFG["r_out"], ax=P["tex_ax"], beta=P["tex_beta"])
        self.sky = SkyP(CFG["fov"], **SKY)
        NW = W * ss
        self.N = W * H * ss * ss
        s = torch.arange(self.N, device=DEV)
        sy = torch.div(s, NW, rounding_mode="floor"); sx = s - sy * NW
        self.px = (sx.float() + 0.5) / ss; self.py = (sy.float() + 0.5) / ss
        order = bayer(ss) if ss & (ss - 1) == 0 else np.random.default_rng(1).permutation(ss * ss).reshape(ss, ss)
        bm = torch.tensor(order, device=DEV, dtype=torch.float32)
        self.sub = (bm[sy % ss, sx % ss] + 0.5) / (ss * ss)   # each sample's position inside the shutter

    def hdr(self, t0, shutter, chunk=2_500_000):
        e = elevation(t0 + shutter / 2)
        out = torch.zeros((self.N, 3), device=DEV)
        for c0 in range(0, self.N, chunk):
            c1 = min(self.N, c0 + chunk)
            buf = lens_rays(self.tab, CFG, e, sample_pos=(self.px[c0:c1], self.py[c0:c1]))
            t = t0 + self.sub[c0:c1] * shutter
            col = torch.zeros((c1 - c0, 3), device=DEV)
            T = torch.ones(c1 - c0, device=DEV)
            for k in range(buf["r"].shape[0]):                 # disk crossings, front to back
                r = buf["r"][k]
                m = ~torch.isnan(r)
                if not m.any():
                    continue
                rgb, a = disk_emission(self.disk, r[m], buf["p"][k][m], buf["lam"][m], P, t[m])
                Tm = T[m]
                col[m] += Tm[:, None] * rgb * (a[:, None] * P["emis_scale"] + (1 - a[:, None]) * P["thin_glow"])
                T[m] = Tm * (1 - a)
            esc = buf["st"] == 2                                # lensed sky, turning with the camera orbit
            rot = TAU / self.sky.fold * (t[esc] / P["loop"]) * P["sky_dir"]
            col[esc] += T[esc, None] * self.sky(buf["dir"][esc], rot)
            out[c0:c1] = col
        ss = CFG["ss"]
        return out.reshape(CFG["H"], ss, CFG["W"], ss, 3).mean(dim=(1, 3))


# ------------------------------------------------------------------ animated typography
def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def gpu(a, ch=False):
    t = torch.tensor(a, device=DEV, dtype=torch.float32)
    return t[..., None] if ch else t


class Overlay:
    """Static type plus: a light sweep across the name, a glow that follows the disk light,
    the tags lighting up in turn, and a blinking terminal cursor."""

    def __init__(self):
        W, H = CFG["W"], CFG["H"]
        blur = lambda a, r: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(r))) / 255.0
        name_grad = ((250, 246, 240), (246, 212, 170))
        base = [
            dict(text=OVERLINE, font=MONO, var=500, size=20, track=4.0, x=X + 2, y=118, color=AMBER),
            dict(text=NAME[0], font=TITLE, size=66, track=7, x=X, y=196, grad=name_grad, glow=0.9),
            dict(text=NAME[1], font=TITLE, size=66, track=7, x=X, y=270, grad=name_grad, glow=0.9),
            dict(text=CAPTION, font=MONO, var=300, size=15, track=0.8, x=X + 2, y=452, color=DIM, alpha=0.95),
        ]
        txt, _ = build_overlay(W, H, base)
        self.base_a, self.base_c = gpu(txt[..., 3] / 255.0, True), gpu(txt[..., :3] / 255.0)
        ntxt, ngm = build_overlay(W, H, base[1:3])
        self.name_a = gpu(ntxt[..., 3] / 255.0)
        self.glow, self.glow_wide, self.glow_tight = (gpu(blur(ngm, r)) for r in (14, 40, 5))
        bb = np.argwhere(ntxt[..., 3] > 0)
        self.name_box = (bb[:, 1].min(), bb[:, 1].max(), bb[:, 0].min(), bb[:, 0].max())
        # one layer per tag so each can light up
        style = dict(font=MONO, var=400, size=19, track=2.0)
        f = font(MONO, 19, 400)
        self.tags, seps, x, allm = [], [], X + 2, txt[..., 3].copy()
        for i, tg in enumerate(TAGS):
            ttxt, _ = build_overlay(W, H, [dict(style, text=tg, x=x, y=338, color=(255, 255, 255))])
            self.tags.append(gpu(ttxt[..., 3] / 255.0, True))
            allm = np.maximum(allm, ttxt[..., 3])
            x += text_width(f, tg, 2.0) + 2.0
            if i < len(TAGS) - 1:
                seps.append(dict(style, text="  ·  ", x=x, y=338, color=(110, 120, 140)))
                x += text_width(f, "  ·  ", 2.0) + 2.0
        stxt, _ = build_overlay(W, H, seps)
        self.sep_a, self.sep_c = gpu(stxt[..., 3] / 255.0, True), gpu(stxt[..., :3] / 255.0)
        self.tag_dim, self.tag_hot = gpu(MUTED) / 255.0, gpu((255, 214, 160)) / 255.0
        cx = X + 2 + text_width(font(MONO, 20, 500), OVERLINE, 4.0) + 10
        ctxt, _ = build_overlay(W, H, [dict(text="▌", font=MONO, var=500, size=20, track=0, x=cx, y=118, color=AMBER)])
        self.cur_a, self.cur_c = gpu(ctxt[..., 3] / 255.0, True), gpu(ctxt[..., :3] / 255.0)
        allm = np.maximum(allm, stxt[..., 3])
        dark = 0.55 * blur(allm, 10)                                   # soft halo + broad scrim behind the type
        sc = blur(allm, 40); sc = np.clip(sc / max(sc.max(), 1e-6), 0, 1)
        self.dark = gpu(1 - (1 - dark) * (1 - 0.35 * sc), True)
        self.yy, self.xx = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                                          torch.arange(W, device=DEV, dtype=torch.float32), indexing="ij")

    def apply(self, y, t, light):
        L = P["loop"]
        y = y * (1 - self.dark)
        g = (0.38 + 0.35 * light) * self.glow + 0.12 * light * self.glow_wide
        y = 1 - (1 - y) * (1 - g[..., None] * gpu((1.0, 0.66, 0.36)))
        x0, x1, y0, y1 = self.name_box
        band = None
        for start in (1.2, 7.2):                                       # light sweep, twice per loop
            ph = (t - start) / 1.6
            if 0 <= ph <= 1:
                cx = x0 - 140 + (x1 - x0 + 280) * smooth(ph)
                d = (self.xx - cx) + (self.yy - (y0 + y1) / 2) * 0.45
                band = torch.exp(-(d / 34) ** 2) * math.sin(math.pi * ph) ** 0.6
        if band is not None:
            halo = (band * (0.9 * self.glow_tight + 0.6 * self.glow))[..., None]
            y = 1 - (1 - y) * (1 - halo * gpu((1.0, 0.78, 0.45)))
        y = y * (1 - self.base_a) + self.base_c * self.base_a
        if band is not None:
            y = y + (band * self.name_a)[..., None] * (1 - y) * 0.9
        seg = L / len(self.tags)
        for i, ta in enumerate(self.tags):                             # tags light up one after another
            u = ((t - i * seg) % L) / seg
            w = smooth(u / 0.18) * (1 - smooth((u - 0.72) / 0.28)) if u < 1 else 0.0
            y = y * (1 - ta) + (self.tag_dim * (1 - w) + self.tag_hot * w) * ta
        y = y * (1 - self.sep_a) + self.sep_c * self.sep_a
        if (t % 1.0) < 0.55:                                           # 1 Hz cursor
            y = y * (1 - self.cur_a) + self.cur_c * self.cur_a
        return y.clamp(0, 1)


# ------------------------------------------------------------------ pipeline
def render_frames(frame_dir, only=None, shutter=0.75):
    R, ov = Renderer(), Overlay()
    n = int(round(P["loop"] * FPS))
    lw = torch.tensor([0.2126, 0.7152, 0.0722], device=DEV)
    t_start, out = time.time(), []
    for i in (range(n) if only is None else only):
        t = i / FPS
        hdr = R.hdr(t, shutter / FPS)
        y = tonemap(hdr, P)
        W, H = CFG["W"], CFG["H"]
        reg = hdr[int(0.40 * H):int(0.70 * H), int(0.30 * W):int(0.52 * W)]      # disk light near the name
        light = min(1.0, max(0.0, float((reg * lw).sum(-1).mean()) * P["exposure"] / 0.6))
        im = Image.fromarray((ov.apply(y, t, light) * 255 + 0.5).clamp(0, 255).byte().cpu().numpy())
        if frame_dir:
            im.save(os.path.join(frame_dir, f"f{i:04d}.png"), compress_level=1)
        else:
            out.append(im)
        if i % 60 == 0:
            print(f"  frame {i}/{n}  elev {elevation(t):+.2f} deg  {time.time()-t_start:.0f}s", flush=True)
    print(f"rendered in {time.time()-t_start:.0f}s")
    return out


def encode_avif(frame_dir, out, crf=34):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(frame_dir, "f%04d.png"), "-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0",
                    "-cpu-used", "4", "-row-mt", "1", "-pix_fmt", "yuv420p10le", "-color_primaries", "bt709",
                    "-color_trc", "iec61966-2-1", "-colorspace", "bt709", "-color_range", "pc", "-loop", "0",
                    "-f", "avif", out], check=True)


def encode_webp(frame_dir, out, width=800, step=2, quality=55):
    """lightweight fallback for browsers without AVIF: smaller, half frame rate."""
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))[::step]
    h = round(CFG["H"] * width / CFG["W"])
    frames = [Image.open(os.path.join(frame_dir, f)).convert("RGB").resize((width, h), Image.LANCZOS) for f in files]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 * step / FPS), loop=0,
                   quality=quality, method=6, minimize_size=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=None, help="render one frame at this time (s) instead")
    a = ap.parse_args()
    ensure_fonts()
    if a.still is not None:
        render_frames(None, only=[int(round(a.still * FPS))])[0].save(os.path.join(ASSETS, "banner-event-horizon-still.png"))
        return
    tmp = tempfile.mkdtemp(prefix="event-horizon-")
    try:
        render_frames(tmp)
        avif = os.path.join(ASSETS, "banner-event-horizon.avif")
        encode_avif(tmp, avif)
        print(f"avif  {os.path.getsize(avif)/1e6:.2f} MB")
        webp = os.path.join(ASSETS, "banner-event-horizon.webp")
        encode_webp(tmp, webp)
        print(f"webp  {os.path.getsize(webp)/1e6:.2f} MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
