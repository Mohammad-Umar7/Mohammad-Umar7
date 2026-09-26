"""
Shader for the traced buffers: accretion-disk emission (blackbody + Doppler/gravitational
redshift, flow-map animated turbulence), lensed procedural sky, blur helpers, ACES curve.
"""
import math, torch, numpy as np
import torch.nn.functional as Fn

DEV = "cuda"
F32 = torch.float32


# ---------------------------------------------------------------- blackbody LUT
def _g(l, mu, s1, s2):
    s = np.where(l < mu, s1, s2)
    return np.exp(-0.5 * ((l - mu) / s) ** 2)


def blackbody_lut(tmin=500.0, tmax=40000.0, n=2048):
    lam = np.arange(360.0, 831.0, 1.0)
    xb = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    Ts = np.exp(np.linspace(np.log(tmin), np.log(tmax), n))
    out = np.zeros((n, 3))
    l = lam * 1e-9
    for i, T in enumerate(Ts):
        B = 1.0 / (l ** 5 * (np.exp(1.4388e-2 / (l * T)) - 1.0))
        X, Y, Z = (B * xb).sum(), (B * yb).sum(), (B * zb).sum()
        rgb = M @ np.array([X, Y, Z]) / Y
        out[i] = np.clip(rgb, 0, None)
    return torch.tensor(out, dtype=F32, device=DEV), math.log(tmin), math.log(tmax)


BB, LTMIN, LTMAX = None, None, None


def bb_rgb(T):
    """chromaticity (Y=1) for temperature tensor T (Kelvin)."""
    global BB, LTMIN, LTMAX
    if BB is None:
        BB, LTMIN, LTMAX = blackbody_lut()
    n = BB.shape[0]
    f = ((torch.log(T.clamp(501, 39999)) - LTMIN) / (LTMAX - LTMIN) * (n - 1))
    i0 = f.floor().long().clamp(0, n - 2); w = (f - i0)[..., None]
    return BB[i0] * (1 - w) + BB[i0 + 1] * w


# ---------------------------------------------------------------- noise
def fft_noise_2d(nh, nw, beta, ax=1.0, ay=1.0, seed=0, lowcut=0.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ky = torch.fft.fftfreq(nh).reshape(-1, 1) * nh
    kx = torch.fft.fftfreq(nw).reshape(1, -1) * nw
    k = torch.sqrt((kx * ax) ** 2 + (ky * ay) ** 2)
    amp = torch.where(k > lowcut, (k + 1e-6) ** (-beta), torch.zeros_like(k))
    amp[0, 0] = 0
    ph = torch.randn(nh, nw, generator=g) + 1j * torch.randn(nh, nw, generator=g)
    n = torch.fft.ifft2(ph * amp).real
    n = (n - n.mean()) / n.std()
    return n.to(DEV, F32)


def fft_noise_3d(n, beta, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k1 = torch.fft.fftfreq(n) * n
    kz, ky, kx = torch.meshgrid(k1, k1, k1, indexing="ij")
    k = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    amp = (k + 1e-6) ** (-beta); amp[0, 0, 0] = 0
    ph = torch.randn(n, n, n, generator=g) + 1j * torch.randn(n, n, n, generator=g)
    v = torch.fft.ifftn(ph * amp).real
    v = (v - v.mean()) / v.std()
    v = torch.cat([v, v[:1]], 0); v = torch.cat([v, v[:, :1]], 1); v = torch.cat([v, v[:, :, :1]], 2)
    return v.to(DEV, F32)[None, None]            # (1,1,n+1,n+1,n+1) periodic-padded


def sample3d(vol, p):
    """periodic trilinear sample of padded volume at points p (N,3), period 1."""
    q = torch.remainder(p, 1.0) * 2 - 1                  # [-1,1]
    grid = q[None, :, None, None, :]                    # (1,N,1,1,3) xyz
    o = Fn.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return o.reshape(-1)


# ---------------------------------------------------------------- sky
class Sky:
    def __init__(self, fov, W, seed=7, mw_normal=(0.25, 0.55, 0.80), mw_strength=1.0,
                 star_layers=((10, 0.55, 0.010, 1.1, 0.6), (26, 0.6, 0.030, 1.3, 0.65), (70, 0.7, 0.10, 1.6, 0.75)),
                 star_gain=1.0, mw_width=0.22, mw_gain=0.030, neb_gain=0.006, neb_color=(0.25, 0.55, 1.0),
                 neb2_gain=0.0, neb2_color=(0.9, 0.35, 0.6), bg=(0.0012, 0.0016, 0.0030)):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.layers = []
        pix = math.radians(fov) / W
        pix_ref = math.radians(fov) / 1600.0
        # (cell size in reference px, existence prob, brightness scale, power, sigma in px)
        for cpx, pe, b0, pw, sp in star_layers:
            N = int(round(2.0 / (cpx * pix_ref)))
            b0 = b0 * star_gain
            R = torch.rand(6, N, N, 5, generator=g)
            ex = (R[..., 4] < pe).float()
            u = R[..., 2].clamp(1e-4, 1)
            b = (b0 * u ** (-pw)).clamp(max=60 * b0) * ex
            temp = 2800 + 9000 * R[..., 3] ** 1.6
            self.layers.append(dict(N=N, ou=R[..., 0].to(DEV) * 0.7 + 0.15, ov=R[..., 1].to(DEV) * 0.7 + 0.15,
                                    b=b.to(DEV), col=bb_rgb(temp.to(DEV)), sig=sp * pix))
        self.vol = fft_noise_3d(96, 1.35, seed=seed + 1)
        n = torch.tensor(mw_normal, dtype=F32, device=DEV); self.mw_n = n / n.norm()
        self.mw_strength = mw_strength
        self.mw_width, self.mw_gain = mw_width, mw_gain
        self.neb_gain, self.neb_color = neb_gain, torch.tensor(neb_color, device=DEV, dtype=F32)
        self.neb2_gain, self.neb2_color = neb2_gain, torch.tensor(neb2_color, device=DEV, dtype=F32)
        self.bg = torch.tensor(bg, device=DEV, dtype=F32)

    def __call__(self, d):
        ax = d.abs(); m = ax.argmax(-1)
        major = torch.gather(ax, 1, m[:, None]).squeeze(1)
        sgn = torch.gather(d, 1, m[:, None]).squeeze(1)
        face = m * 2 + (sgn < 0).long()
        ui = torch.where(m == 0, 1, 0); vi = torch.where(m == 2, 1, 2)
        u = torch.gather(d, 1, ui[:, None]).squeeze(1) / major
        v = torch.gather(d, 1, vi[:, None]).squeeze(1) / major
        col = torch.zeros_like(d)
        for L in self.layers:
            N = L["N"]
            i = ((u + 1) * 0.5 * N).floor().long().clamp(0, N - 1)
            j = ((v + 1) * 0.5 * N).floor().long().clamp(0, N - 1)
            s2 = 2 * L["sig"] ** 2
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii = (i + di).clamp(0, N - 1); jj = (j + dj).clamp(0, N - 1)
                    su = -1 + 2 * (ii.float() + L["ou"][face, ii, jj]) / N
                    sv = -1 + 2 * (jj.float() + L["ov"][face, ii, jj]) / N
                    d2 = (u - su) ** 2 + (v - sv) ** 2
                    w = L["b"][face, ii, jj] * torch.exp(-d2 / s2)
                    col += w[:, None] * L["col"][face, ii, jj]
        # milky-way band with dust lanes
        h = (d * self.mw_n).sum(-1)
        band = torch.exp(-(h / self.mw_width) ** 2)
        p = d * 0.9
        n1 = sample3d(self.vol, p * 0.55 + 0.13)
        n2 = sample3d(self.vol, p * 1.7 + 0.61)
        n3 = sample3d(self.vol, p * 4.1 + 0.37)
        clouds = (0.55 + 0.30 * n1 + 0.15 * n2 + 0.08 * n3).clamp(min=0)
        dust = torch.sigmoid((n2 * 0.7 + n3 * 0.5 - 0.2) * 3.0)
        mw = band * clouds * (1 - 0.85 * dust) * self.mw_strength
        core = torch.exp(-(h / 0.09) ** 2)
        warm = torch.tensor([1.0, 0.78, 0.58], device=DEV); cool = torch.tensor([0.55, 0.68, 1.0], device=DEV)
        mwcol = (warm * core[:, None] + cool * (1 - core[:, None]))
        col += self.mw_gain * mw[:, None] * mwcol
        # faint nebular tints away from the band
        neb = (sample3d(self.vol, p * 0.8 + 0.77) * 0.5 + 0.5).clamp(0, 1) ** 3
        col += self.neb_gain * neb[:, None] * self.neb_color
        if self.neb2_gain > 0:
            nb2 = (sample3d(self.vol, p * 1.1 + 0.29) * 0.5 + 0.45).clamp(0, 1) ** 4
            col += self.neb2_gain * nb2[:, None] * self.neb2_color
        col += self.bg
        return col


# ---------------------------------------------------------------- disk
class Disk:
    def __init__(self, r_in, r_out, seed=3, NU=1024, NP=4096, ax=9.0, beta=1.15, hot_beta=1.6, hot_ax=4.0, seedB=None):
        self.r_in, self.r_out = r_in, r_out
        self.lu0, self.lu1 = math.log(r_in), math.log(r_out)
        self.NU, self.NP = NU, NP
        # two decorrelated turbulent layers (flow-map), elongated azimuthally
        self.texA = fft_noise_2d(NU, NP, beta=beta, ax=ax, ay=1.0, seed=seed)
        sB = seed + 11 if seedB is None else seedB
        self.texB = fft_noise_2d(NU, NP, beta=beta, ax=ax, ay=1.0, seed=sB)
        # clumpy hot-spot layers (coarser, less elongated)
        self.hotA = fft_noise_2d(NU, NP, beta=hot_beta, ax=hot_ax, ay=1.0, seed=seed + 21)
        self.hotB = fft_noise_2d(NU, NP, beta=hot_beta, ax=hot_ax, ay=1.0, seed=sB + 20)
        # static ringlets (radial only)
        self.rings = fft_noise_2d(NU, 8, beta=0.9, ax=1.0, ay=1.0, seed=seed + 5)[:, 0]
        self.pad = {}

    def _padded(self, tex):
        k = id(tex)
        if k not in self.pad:
            self.pad[k] = torch.cat([tex, tex[:, :1]], 1)[None, None]
        return self.pad[k]

    def sample(self, tex, r, phi):
        u = (torch.log(r) - self.lu0) / (self.lu1 - self.lu0) * 2 - 1        # [-1,1] rows
        pw = torch.remainder(phi, 2 * math.pi) / (2 * math.pi)             # [0,1)
        grid = torch.stack([pw * 2 - 1, u], -1)[None, :, None, :]
        return Fn.grid_sample(self._padded(tex), grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)

    def ring(self, r):
        u = (torch.log(r) - self.lu0) / (self.lu1 - self.lu0) * (self.NU - 1)
        i0 = u.floor().long().clamp(0, self.NU - 2); w = u - i0
        return self.rings[i0] * (1 - w) + self.rings[i0 + 1] * w


# ---------------------------------------------------------------- frame
def disk_emission(disk, r, phi, lam, P, t):
    """returns (rgb, alpha) for crossings at radius r, azimuth phi."""
    r_in, r_out = disk.r_in, disk.r_out
    Om = r ** -1.5
    g = torch.sqrt((1 - 3.0 / r).clamp(min=1e-4)) / (1 - Om * lam)
    # turbulent texture, flow-map animated with variance-preserving blend
    loop = P["loop"]
    ph = (t / loop) % 1.0
    wA = math.sin(math.pi * ph) ** 2; wB = 1 - wA
    nrm = math.sqrt(wA * wA + wB * wB)
    offA = P["spin"] * ph; offB = P["spin"] * ((ph + 0.5) % 1.0)
    shear = P["shear"]
    pA = phi - Om * (offA + shear); pB = phi - Om * (offB + shear) + 1.3
    n = (wA * disk.sample(disk.texA, r, pA) + wB * disk.sample(disk.texB, r, pB)) / nrm
    hs = (wA * disk.sample(disk.hotA, r, pA) + wB * disk.sample(disk.hotB, r, pB)) / nrm
    rg = disk.ring(r)
    tex = n * P["turb"] + rg * P["ringamp"]
    fil = torch.sigmoid(tex * P["contrast"] + P["bias"])
    hot = torch.sigmoid((hs - P.get("hot_thr", 1.2)) * 4.0) * P.get("hot_amt", 0.0)
    # radial profiles
    rie = P["r_in_eff"]
    f = (r ** -3) * (1 - torch.sqrt(rie / r)).clamp(min=0)
    rp = (49 / 36) * rie
    fpk = rp ** -3 * (1 - math.sqrt(rie / rp))
    fr = (f / fpk).clamp(min=0)
    Tr = P["T0"] * fr ** P.get("Texp", 0.25)
    edge_in = torch.sigmoid((r - r_in) / 0.08)
    rag = P.get("ragged", 0.0) * (n * 0.6 + hs * 0.4)
    edge_out = (1 - torch.sigmoid((r - (r_out - P["fade_w"] * 0.5) + rag * P["fade_w"] * 0.5) / (P["fade_w"] * 0.18)))
    # local temperature follows local brightness (dim lanes cooler/redder, bright streaks hotter)
    tl = P.get("t_lo", 0.72)
    Tloc = Tr * (tl + (1 - tl) * fil + 0.25 * hot)
    Tobs = Tloc * g ** P["gT"]
    I = fr ** P["Ipow"] * g ** P["gI"] * P["gain"]
    amp = (P["base"] + (1 - P["base"]) * fil) * (1 + hot * 2.0)
    col = bb_rgb(Tobs.clamp(min=600)) * (I * amp * edge_out * edge_in)[:, None]
    dens = P["alpha"] * (P.get("a_lo", 0.55) + (1 - P.get("a_lo", 0.55)) * fil) * (1 + P.get("a_inner", 0.0) * fr)
    alpha = (dens * edge_out * edge_in).clamp(0, 1)
    return col, alpha


def shade(buf, disk, sky, P, t, device=DEV, skycache=None):
    cfg = buf["cfg"]; W, H, ss = cfg["W"], cfg["H"], cfg["ss"]
    N = W * H * ss * ss
    out = torch.zeros((N, 3), device=device)
    K = buf["r"].shape[0]
    chunk = 2_000_000
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        lam = buf["lam"][c0:c1].to(device)
        st = buf["st"][c0:c1].to(device)
        col = torch.zeros((c1 - c0, 3), device=device)
        T = torch.ones(c1 - c0, device=device)
        for k in range(K):
            r = buf["r"][k, c0:c1].to(device)
            m = ~torch.isnan(r)
            if not m.any():
                continue
            rgb, a = disk_emission(disk, r[m], buf["p"][k, c0:c1].to(device)[m], lam[m], P, t)
            Tm = T[m]
            col[m] += Tm[:, None] * rgb * a[:, None] * P["emis_scale"] + Tm[:, None] * rgb * (1 - a[:, None]) * P["thin_glow"]
            T[m] = Tm * (1 - a)
        if skycache is not None:
            skyc = skycache[c0:c1].to(device)
        else:
            skyc = torch.zeros((c1 - c0, 3), device=device)
            e = st == 2
            if e.any():
                skyc[e] = sky(buf["dir"][c0:c1].to(device)[e])
        col += T[:, None] * skyc
        out[c0:c1] = col
    img = out.reshape(H, ss, W, ss, 3).mean(dim=(1, 3))
    return img


def sky_cache(buf, sky, device=DEV):
    N = buf["st"].shape[0]
    out = torch.zeros((N, 3), dtype=torch.float16)
    chunk = 2_000_000
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        st = buf["st"][c0:c1].to(device)
        skyc = torch.zeros((c1 - c0, 3), device=device)
        e = st == 2
        if e.any():
            skyc[e] = sky(buf["dir"][c0:c1].to(device)[e])
        out[c0:c1] = skyc.half().cpu()
    return out


# ---------------------------------------------------------------- post
def gauss_blur(img, sigma, axis="xy"):
    """img (H,W,3) on GPU; separable gaussian with reflect padding; big sigma via downsample."""
    x = img.permute(2, 0, 1)[None]
    scale = 1
    while sigma / scale > 24:
        scale *= 2
    if scale > 1:
        x = Fn.avg_pool2d(x, scale, ceil_mode=True) if axis == "xy" else Fn.avg_pool2d(x, (1, scale), ceil_mode=True)
    s = sigma / scale
    rad = int(math.ceil(3 * s))
    k = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, device=img.device, dtype=F32) / s) ** 2); k /= k.sum()
    C = 3
    if axis in ("xy", "x"):
        x = Fn.pad(x, (rad, rad, 0, 0), mode="reflect" if rad < x.shape[3] else "replicate")
        x = Fn.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    if axis in ("xy", "y"):
        x = Fn.pad(x, (0, 0, rad, rad), mode="reflect" if rad < x.shape[2] else "replicate")
        x = Fn.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    if scale > 1:
        x = Fn.interpolate(x, size=img.shape[:2], mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0)


def aces(x):
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
