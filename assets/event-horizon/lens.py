"""
Fast Schwarzschild lensing via a precomputed orbit table.

Every photon moves in a plane through the hole, and in that plane its path depends only on the
impact parameter b:   u'' + u = 3 M u^2   (u = 1/r, ' = d/dphi, M = 1).
So we integrate u(phi; b) once for a dense set of b (float64), then for any camera pose each
pixel needs only: its b, its orbital plane, where that plane cuts the disk (the line of nodes),
and table lookups at the node angles phi_1 + k*pi.  Milliseconds per frame instead of seconds.
"""
import math, torch

DEV = "cuda"
F32 = torch.float32
BC = 3 * math.sqrt(3.0)


class OrbitTable:
    def __init__(self, D, b_max, nb_uni=8192, nb_log=2048, dphi=0.004, phi_max=6 * math.pi):
        self.D, self.dphi = D, dphi
        b_uni = torch.linspace(1e-3, b_max, nb_uni, dtype=torch.float64)
        t = torch.linspace(0.25, 6.8, nb_log, dtype=torch.float64)
        b = torch.cat([b_uni, BC - 10 ** (-t), BC + 10 ** (-t)])
        b = torch.unique(b[(b > 0) & (b <= b_max)])
        self.b = b.to(DEV)
        nb = b.numel()
        ns = int(math.ceil(phi_max / dphi))
        self.ns = ns
        u0 = 1.0 / D
        bb = b.to(DEV)
        u = torch.full((nb,), u0, dtype=torch.float64, device=DEV)
        w = torch.sqrt((1 / bb ** 2 - u0 ** 2 * (1 - 2 * u0)).clamp(min=0))
        U = torch.empty((nb, ns + 1), dtype=F32, device=DEV)
        phi_end = torch.full((nb,), phi_max, dtype=torch.float64, device=DEV)
        alive = torch.ones(nb, dtype=torch.bool, device=DEV)
        f = lambda u: -u + 3 * u * u
        h = dphi
        for i in range(ns):
            U[:, i] = u.float()
            k1u = w;               k1w = f(u)
            k2u = w + 0.5 * h * k1w; k2w = f(u + 0.5 * h * k1u)
            k3u = w + 0.5 * h * k2w; k3w = f(u + 0.5 * h * k2u)
            k4u = w + h * k3w;       k4w = f(u + h * k3u)
            un = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
            wn = w + h / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
            esc = alive & (un <= 0)
            cap = alive & (un >= 0.5)
            if esc.any():
                phi_end[esc] = i * h + h * (u[esc] / (u[esc] - un[esc]))
            if cap.any():
                phi_end[cap] = i * h + h * ((0.5 - u[cap]) / (un[cap] - u[cap]))
            alive = alive & ~esc & ~cap
            u = torch.where(alive, un, torch.where(un <= 0, torch.zeros_like(un), torch.full_like(un, 0.5)))
            w = torch.where(alive, wn, torch.zeros_like(wn))
        U[:, ns] = u.float()
        self.U = U
        self.phi_end = phi_end.float()
        self.bf = self.b.float()

    def _bidx(self, b):
        bd = b.double().contiguous()
        ib = (torch.searchsorted(self.b, bd) - 1).clamp(0, self.b.numel() - 2)
        b0 = self.b[ib]; b1 = self.b[ib + 1]          # float64: grid points near b_c collide in float32
        wb = ((bd - b0) / (b1 - b0).clamp(min=1e-15)).clamp(0, 1).float()
        return ib, wb

    def end(self, ib, wb):
        return self.phi_end[ib] * (1 - wb) + self.phi_end[ib + 1] * wb

    def u_at(self, ib, wb, phi):
        fp = (phi / self.dphi).clamp(0, self.ns - 1e-3)
        ip = fp.floor().long(); wp = fp - ip
        U = self.U
        a = U[ib, ip] * (1 - wp) + U[ib, ip + 1] * wp
        c = U[ib + 1, ip] * (1 - wp) + U[ib + 1, ip + 1] * wp
        return a * (1 - wb) + c * wb


def camera_basis(D, elev, azim, roll=0.0):
    e, a, rl = math.radians(elev), math.radians(azim), math.radians(roll)
    cam = torch.tensor([D * math.cos(e) * math.cos(a), D * math.cos(e) * math.sin(a), D * math.sin(e)], dtype=torch.float64)
    fwd = -cam / cam.norm()
    zup = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.linalg.cross(fwd, zup); right /= right.norm()
    up = torch.linalg.cross(right, fwd)
    right, up = right * math.cos(rl) + up * math.sin(rl), -right * math.sin(rl) + up * math.cos(rl)
    return cam, fwd, right, up


def lens_rays(tab, cfg, elev, K=4, sample_pos=None):
    """Returns the same buffer layout as trace.trace() for camera elevation `elev` (deg).
    sample_pos: optional (px, py) tensors (pixel coords) to trace instead of the regular grid."""
    W, H, ss, D = cfg["W"], cfg["H"], cfg["ss"], cfg["D"]
    cam, fwd, right, up = camera_basis(D, elev, cfg["azim"], cfg.get("roll", 0.0))
    f = (W / 2) / math.tan(math.radians(cfg["fov"]) / 2)
    cx, cy = cfg["cx"] * W, cfg["cy"] * H
    if sample_pos is None:
        NW, NH = W * ss, H * ss
        s = torch.arange(NW * NH, device=DEV)
        sy = torch.div(s, NW, rounding_mode="floor"); sx = s - sy * NW
        px = (sx.float() + 0.5) / ss; py = (sy.float() + 0.5) / ss
    else:
        px, py = sample_pos
    camg, fwdg, rightg, upg = (t.to(DEV, F32) for t in (cam, fwd, right, up))
    v = fwdg * f + (px - cx)[:, None] * rightg + (cy - py)[:, None] * upg
    v = v / v.norm(dim=-1, keepdim=True)
    e1 = camg / D
    ve1 = (v * e1).sum(-1)
    perp = v - ve1[:, None] * e1
    sn = perp.norm(dim=-1).clamp(min=1e-9)
    e2 = perp / sn[:, None]
    b = D * sn
    lam = -(camg[0] * v[:, 1] - camg[1] * v[:, 0])
    n = torch.linalg.cross(e1.expand_as(e2), e2)
    zhat = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    L = torch.linalg.cross(n, zhat.expand_as(n))
    L = L / L.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    phi_node = torch.atan2((L * e2).sum(-1), (L * e1).sum(-1))
    phi1 = torch.remainder(phi_node, math.pi)
    ib, wb = tab._bidx(b)
    pend = tab.end(ib, wb)
    captured = b < BC
    N = b.numel()
    hr = torch.full((K, N), float("nan"), device=DEV)
    hp = torch.full((K, N), float("nan"), device=DEV)
    for k in range(K):
        phk = phi1 + k * math.pi
        ok = phk < pend
        u = tab.u_at(ib, wb, phk)
        r = 1.0 / u.clamp(min=1e-6)
        ok = ok & (r >= cfg["r_in"]) & (r <= cfg["r_out"])
        x = r[:, None] * (torch.cos(phk)[:, None] * e1 + torch.sin(phk)[:, None] * e2)
        hr[k] = torch.where(ok, r, torch.full_like(r, float("nan")))
        hp[k] = torch.where(ok, torch.atan2(x[:, 1], x[:, 0]), torch.full_like(r, float("nan")))
    d = torch.cos(pend)[:, None] * e1 + torch.sin(pend)[:, None] * e2
    st = torch.where(captured, torch.ones_like(b, dtype=torch.int8), torch.full_like(b, 2, dtype=torch.int8))
    return dict(r=hr, p=hp, dir=d, st=st, lam=lam, cfg=cfg)
