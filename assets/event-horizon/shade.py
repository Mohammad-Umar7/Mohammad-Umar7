"""
Animated shader: Keplerian flow-map disk + orbiting hot-spot flares (Doppler-beamed, lensed),
a 6-fold periodic sky that rotates about the spin axis (camera orbit), and per-sample
time offsets inside each pixel for motion blur.
"""
import math, torch, numpy as np
import torch.nn.functional as Fn

DEV = "cuda"
F32 = torch.float32
TAU = 2 * math.pi


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
        out[i] = np.clip(M @ np.array([X, Y, Z]) / Y, 0, None)
    return torch.tensor(out, dtype=F32, device=DEV), math.log(tmin), math.log(tmax)


BB, LTMIN, LTMAX = None, None, None


def bb_rgb(T):
    global BB, LTMIN, LTMAX
    if BB is None:
        BB, LTMIN, LTMAX = blackbody_lut()
    n = BB.shape[0]
    f = ((torch.log(T.clamp(501, 39999)) - LTMIN) / (LTMAX - LTMIN) * (n - 1))
    i0 = f.floor().long().clamp(0, n - 2); w = (f - i0)[..., None]
    return BB[i0] * (1 - w) + BB[i0 + 1] * w


# ---------------------------------------------------------------- noise
def fft_noise_2d(nh, nw, beta, ax=1.0, ay=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ky = torch.fft.fftfreq(nh).reshape(-1, 1) * nh
    kx = torch.fft.fftfreq(nw).reshape(1, -1) * nw
    k = torch.sqrt((kx * ax) ** 2 + (ky * ay) ** 2)
    amp = (k + 1e-6) ** (-beta); amp[0, 0] = 0
    ph = torch.randn(nh, nw, generator=g) + 1j * torch.randn(nh, nw, generator=g)
    n = torch.fft.ifft2(ph * amp).real
    return ((n - n.mean()) / n.std()).to(DEV, F32)


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
    return v.to(DEV, F32)[None, None]


def sample3d(vol, p):
    q = torch.remainder(p, 1.0) * 2 - 1
    o = Fn.grid_sample(vol, q[None, :, None, None, :], mode="bilinear", padding_mode="border", align_corners=True)
    return o.reshape(-1)


# ---------------------------------------------------------------- periodic rotating sky
class SkyP:
    """Sky that is exactly `fold`-fold symmetric about the z (spin) axis, so rotating it by
    2*pi/fold over the loop gives a seamless camera orbit."""

    def __init__(self, fov, fold=6, seed=7,
                 star_layers=((14, 0.5, 0.008, 1.1, 0.6), (34, 0.55, 0.025, 1.3, 0.65), (90, 0.7, 0.08, 1.6, 0.75)),
                 star_gain=1.0, neb_gain=0.010, neb_color=(0.25, 0.5, 1.0), neb2_gain=0.006, neb2_color=(0.85, 0.4, 0.75),
                 band_gain=0.035, band_z=0.0, band_w=0.18, bg=(0.0012, 0.0016, 0.0030), W_ref=1600):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.fold = fold; self.per = TAU / fold
        pix_ref = math.radians(fov) / W_ref
        self.layers = []
        for cpx, pe, b0, pw, sp in star_layers:
            ca = cpx * pix_ref
            Np = max(4, int(round(self.per / ca))); Nz = max(4, int(round(2.0 / ca)))
            R = torch.rand(Np, Nz, 5, generator=g)
            ex = (R[..., 4] < pe).float()
            b = (b0 * star_gain * R[..., 2].clamp(1e-4, 1) ** (-pw)).clamp(max=60 * b0 * star_gain) * ex
            temp = 2800 + 9000 * R[..., 3] ** 1.6
            self.layers.append(dict(Np=Np, Nz=Nz, ou=(R[..., 0] * 0.8 + 0.1).to(DEV), ov=(R[..., 1] * 0.8 + 0.1).to(DEV),
                                    b=b.to(DEV), col=bb_rgb(temp.to(DEV)), sig=sp * pix_ref))
        self.vol = fft_noise_3d(96, 1.35, seed=seed + 1)
        self.neb_gain, self.neb2_gain, self.band_gain = neb_gain, neb2_gain, band_gain
        self.neb_color = torch.tensor(neb_color, device=DEV); self.neb2_color = torch.tensor(neb2_color, device=DEV)
        self.band_z, self.band_w = band_z, band_w
        self.bg = torch.tensor(bg, device=DEV)

    def __call__(self, d, rot):
        c, s = torch.cos(rot), torch.sin(rot)
        x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c; z = d[:, 2].clamp(-1, 1)
        psi = torch.remainder(torch.atan2(y, x), TAU)
        tile = torch.floor(psi / self.per); psl = psi - tile * self.per
        col = torch.zeros((d.shape[0], 3), device=d.device)
        for L in self.layers:
            Np, Nz = L["Np"], L["Nz"]
            i = torch.floor(psl / self.per * Np).long().clamp(0, Np - 1)
            j = torch.floor((z + 1) * 0.5 * Nz).long().clamp(0, Nz - 1)
            s2 = 2 * L["sig"] ** 2
            for di in (-1, 0, 1):
                iu = i + di                                   # unwrapped cell index within this tile
                ii = torch.remainder(iu, Np)
                for dj in (-1, 0, 1):
                    jj = (j + dj).clamp(0, Nz - 1)
                    ps = (tile * Np + iu.float() + L["ou"][ii, jj]) * (self.per / Np)
                    zs = -1 + 2 * (jj.float() + L["ov"][ii, jj]) / Nz
                    rs = torch.sqrt((1 - zs * zs).clamp(min=0))
                    d2 = (x - rs * torch.cos(ps)) ** 2 + (y - rs * torch.sin(ps)) ** 2 + (z - zs) ** 2
                    w = L["b"][ii, jj] * torch.exp(-d2 / s2)
                    col += w[:, None] * L["col"][ii, jj]
        # fold-periodic diffuse structure
        a = 1.3
        # circle radius in noise space chosen so horizontal and vertical feature scales match
        rr = a * 0.25
        p = torch.stack([torch.cos(self.fold * psi) * rr, torch.sin(self.fold * psi) * rr, z * rr * self.fold], -1)
        n1 = sample3d(self.vol, p * 1.0 + 0.13); n2 = sample3d(self.vol, p * 2.3 + 0.61); n3 = sample3d(self.vol, p * 5.1 + 0.37)
        clouds = (0.55 + 0.30 * n1 + 0.15 * n2 + 0.08 * n3).clamp(min=0)
        dust = torch.sigmoid((n2 * 0.7 + n3 * 0.5 - 0.2) * 3.0)
        band = torch.exp(-((z - self.band_z) / self.band_w) ** 2)
        warm = torch.tensor([1.0, 0.8, 0.62], device=d.device); cool = torch.tensor([0.55, 0.68, 1.0], device=d.device)
        core = torch.exp(-((z - self.band_z) / (self.band_w * 0.4)) ** 2)
        col += self.band_gain * (band * clouds * (1 - 0.85 * dust))[:, None] * (warm * core[:, None] + cool * (1 - core[:, None]))
        neb = (n1 * 0.5 + 0.5).clamp(0, 1) ** 3
        col += self.neb_gain * neb[:, None] * self.neb_color
        nb2 = (sample3d(self.vol, p * 1.6 + 0.29) * 0.5 + 0.45).clamp(0, 1) ** 4
        col += self.neb2_gain * nb2[:, None] * self.neb2_color
        return col + self.bg


# ---------------------------------------------------------------- disk
class Disk:
    def __init__(self, r_in, r_out, seed=7, seedB=19, NU=1024, NP=4096, ax=14.0, beta=1.0, hot_beta=1.6, hot_ax=4.0):
        self.r_in, self.r_out = r_in, r_out
        self.lu0, self.lu1 = math.log(r_in), math.log(r_out)
        self.NU = NU
        self.texA = fft_noise_2d(NU, NP, beta=beta, ax=ax, seed=seed)
        self.texB = fft_noise_2d(NU, NP, beta=beta, ax=ax, seed=seedB)
        self.hotA = fft_noise_2d(NU, NP, beta=hot_beta, ax=hot_ax, seed=seed + 21)
        self.hotB = fft_noise_2d(NU, NP, beta=hot_beta, ax=hot_ax, seed=seedB + 20)
        self.rings = fft_noise_2d(NU, 8, beta=0.9, seed=seed + 5)[:, 0]
        self.pad = {}

    def _padded(self, tex):
        k = id(tex)
        if k not in self.pad:
            self.pad[k] = torch.cat([tex, tex[:, :1]], 1)[None, None]
        return self.pad[k]

    def sample(self, tex, r, phi):
        u = (torch.log(r) - self.lu0) / (self.lu1 - self.lu0) * 2 - 1
        pw = torch.remainder(phi, TAU) / TAU
        grid = torch.stack([pw * 2 - 1, u], -1)[None, :, None, :]
        return Fn.grid_sample(self._padded(tex), grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)

    def ring(self, r):
        u = (torch.log(r) - self.lu0) / (self.lu1 - self.lu0) * (self.NU - 1)
        i0 = u.floor().long().clamp(0, self.NU - 2); w = u - i0
        return self.rings[i0] * (1 - w) + self.rings[i0 + 1] * w


def flare_field(r, phi, t, P):
    """sum of orbiting hot spots; each completes an integer number of orbits per loop."""
    tot = torch.zeros_like(r)
    for (rk, nk, ph0, amp, s_r, s_lead, s_tail, drift) in P["flares"]:
        phk = ph0 + TAU * nk * t / P["loop"]
        dphi = torch.remainder(phi - phk + math.pi, TAU) - math.pi          # (-pi, pi], + = ahead
        behind = (-dphi).clamp(min=0)
        rc = rk + drift * behind
        radial = torch.exp(-((r - rc) / (s_r * (1 + 0.8 * behind))) ** 2)
        # tail fades to exactly zero before it wraps round to the far side (no seam at dphi = +-pi)
        az = torch.where(dphi > 0, torch.exp(-(dphi / s_lead) ** 2),
                         torch.exp(-behind / s_tail) * (1 - behind / math.pi).clamp(min=0) ** 2)
        tot = tot + amp * radial * az
    return tot


def disk_emission(disk, r, phi, lam, P, t):
    """t: per-sample time (seconds). returns (rgb, alpha)."""
    r_in, r_out = disk.r_in, disk.r_out
    Om = r ** -1.5
    g = torch.sqrt((1 - 3.0 / r).clamp(min=1e-4)) / (1 - Om * lam)
    Tf = P["loop"] / P["flow_cycles"]
    ph = torch.remainder(t / Tf, 1.0)
    wA = torch.sin(math.pi * ph) ** 2; wB = 1 - wA
    nrm = torch.sqrt(wA * wA + wB * wB)
    rate = P["rate"]
    offA = rate * Tf * ph; offB = rate * Tf * torch.remainder(ph + 0.5, 1.0)
    shear = P["shear"]
    pA = phi - Om * (offA + shear); pB = phi - Om * (offB + shear) + 1.3
    n = (wA * disk.sample(disk.texA, r, pA) + wB * disk.sample(disk.texB, r, pB)) / nrm
    hs = (wA * disk.sample(disk.hotA, r, pA) + wB * disk.sample(disk.hotB, r, pB)) / nrm
    tex = n * P["turb"] + disk.ring(r) * P["ringamp"]
    fil = torch.sigmoid(tex * P["contrast"] + P["bias"])
    hot = torch.sigmoid((hs - P["hot_thr"]) * 4.0) * P["hot_amt"]
    fl = flare_field(r, phi, t, P) if P.get("flares") else torch.zeros_like(r)
    if P.get("arm_amp", 0) > 0:
        m_arm = P["arm_m"]
        pat = TAU / m_arm * (t / P["loop"]) * P.get("arm_turns", 1)
        psi_arm = m_arm * (phi - pat) + P["arm_wind"] * torch.log(r)
        arm = 0.5 + 0.5 * torch.cos(psi_arm)
        arm = arm ** P.get("arm_sharp", 3.0)
        fil = fil * (1 - P["arm_amp"] + P["arm_amp"] * (0.35 + 1.3 * arm))
    rie = P["r_in_eff"]
    f = (r ** -3) * (1 - torch.sqrt(rie / r)).clamp(min=0)
    rp = (49 / 36) * rie
    fpk = rp ** -3 * (1 - math.sqrt(rie / rp))
    fr = (f / fpk).clamp(min=0)
    Tr = P["T0"] * fr ** P["Texp"]
    edge_in = torch.sigmoid((r - r_in) / 0.08)
    rag = P["ragged"] * (n * 0.6 + hs * 0.4)
    edge_out = 1 - torch.sigmoid((r - (r_out - P["fade_w"] * 0.5) + rag * P["fade_w"] * 0.5) / (P["fade_w"] * 0.18))
    tl = P["t_lo"]
    flw = fl / (1 + fl)
    Tloc = Tr * (tl + (1 - tl) * fil + 0.25 * hot) * (1 + P["flare_heat"] * flw)
    Tobs = Tloc * g ** P["gT"]
    I = fr ** P["Ipow"] * g ** P["gI"] * P["gain"]
    amp = (P["base"] + (1 - P["base"]) * fil) * (1 + hot * 2.0) * (1 + fl)
    col = bb_rgb(Tobs.clamp(min=600)) * (I * amp * edge_out * edge_in)[:, None]
    dens = P["alpha"] * (P["a_lo"] + (1 - P["a_lo"]) * fil) * (1 + P["a_inner"] * fr) + flw * 0.5
    alpha = (dens * edge_out * edge_in).clamp(0, 1)
    return col, alpha


# ---------------------------------------------------------------- frame helpers
def bayer(n):
    if n == 1:
        return np.zeros((1, 1))
    m = bayer(n // 2)
    return np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])


# ---------------------------------------------------------------- post helpers
def gauss_blur(img, sigma, axis="xy"):
    x = img.permute(2, 0, 1)[None]
    scale = 1
    while sigma / scale > 24:
        scale *= 2
    if scale > 1:
        x = Fn.avg_pool2d(x, scale, ceil_mode=True) if axis == "xy" else Fn.avg_pool2d(x, (1, scale), ceil_mode=True)
    s = sigma / scale
    rad = int(math.ceil(3 * s))
    k = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, device=img.device, dtype=F32) / s) ** 2); k /= k.sum()
    if axis in ("xy", "x"):
        x = Fn.pad(x, (rad, rad, 0, 0), mode="reflect" if rad < x.shape[3] else "replicate")
        x = Fn.conv2d(x, k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    if axis in ("xy", "y"):
        x = Fn.pad(x, (0, 0, rad, rad), mode="reflect" if rad < x.shape[2] else "replicate")
        x = Fn.conv2d(x, k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    if scale > 1:
        x = Fn.interpolate(x, size=img.shape[:2], mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0)


def aces(x):
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)


def tonemap(hdr, P):
    H, W = hdr.shape[:2]
    x = hdr * P["exposure"]
    glow = torch.zeros_like(x)
    for s, w in P["bloom"]:
        glow += w * gauss_blur(x, s * W / 1600)
    x = x + glow
    br = (x - P["streak_thr"]).clamp(min=0)
    br = br / (1 + br / P.get("streak_cap", 1e9))                        # soft-cap so flares don't draw hard lines
    streak = gauss_blur(gauss_blur(br, P["streak_sigma"] * W / 1600, axis="x"), P.get("streak_sy", 1.2) * W / 1600, axis="y")
    x = x + streak * torch.tensor(P["streak_tint"], device=x.device) * P["streak_amt"]
    yy = torch.linspace(-1, 1, H, device=x.device)[:, None]; xx = torch.linspace(-1, 1, W, device=x.device)[None, :]
    x = x * (1 - P["vignette"] * ((xx * 0.85) ** 2 + (yy * 0.6) ** 2).clamp(0, 1))[..., None]
    lw = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device)
    lum = (x * lw).sum(-1, keepdim=True)
    x = (lum + (x - lum) * P["sat"]).clamp(min=0)
    y1 = aces(x).clamp(0, 1)
    L = (x * lw).sum(-1, keepdim=True).clamp(min=1e-6)
    y2 = x * (aces(L) / L)
    mx = y2.max(-1, keepdim=True).values
    y2 = torch.where(mx > 1, y2 / mx + (1 - 1 / mx) * 0.9, y2).clamp(0, 1)
    y = y1 * (1 - P["hue_preserve"]) + y2 * P["hue_preserve"]
    y = torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
    return y.clamp(0, 1)
