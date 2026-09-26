"""
Schwarzschild black-hole geodesic ray tracer (GPU, PyTorch).

Units: G = c = M = 1  ->  event horizon r = 2, photon sphere r = 3, ISCO r = 6.
Null geodesics are integrated with the exact orbit-shape equation
    x'' = -3 M h^2 x / |x|^5,   h = |x cross x'|   (conserved)
which reproduces u'' + u = 3 M u^2 in each ray's orbital plane.

Output: per-sample buffers (disk crossings, escape direction, status, L_z)
that the shader re-uses for every animation frame.
"""
import math, time, torch, numpy as np

DEV = "cuda"
F32 = torch.float32


def camera(cfg):
    W, H = cfg["W"], cfg["H"]
    e, a, rl = (math.radians(cfg[k]) for k in ("elev", "azim", "roll"))
    D = cfg["D"]
    cam = torch.tensor([D * math.cos(e) * math.cos(a),
                        D * math.cos(e) * math.sin(a),
                        D * math.sin(e)], dtype=torch.float64)
    fwd = -cam / cam.norm()
    zup = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.linalg.cross(fwd, zup); right /= right.norm()
    up = torch.linalg.cross(right, fwd)
    right, up = (right * math.cos(rl) + up * math.sin(rl),
                 -right * math.sin(rl) + up * math.cos(rl))
    # optional aim offset: rotate the view so the hole sits off-centre
    f = (W / 2) / math.tan(math.radians(cfg["fov"]) / 2)
    cx, cy = cfg["cx"] * W, cfg["cy"] * H
    return cam, fwd, right, up, f, cx, cy


def accel(x, h2):
    r2 = (x * x).sum(-1, keepdim=True)
    return -3.0 * h2 * x / (r2 * r2 * torch.sqrt(r2))


def trace(cfg, K=4, chunk=3_000_000, max_steps=3000, k_step=0.012, verbose=True):
    W, H, ss = cfg["W"], cfg["H"], cfg["ss"]
    r_in, r_out = cfg["r_in"], cfg["r_out"]
    cam, fwd, right, up, f, cx, cy = camera(cfg)
    NW, NH = W * ss, H * ss
    N = NW * NH
    R_esc = cfg["D"] * 1.02 + 20.0

    out_r = torch.full((K, N), float("nan"), dtype=F32)
    out_p = torch.full((K, N), float("nan"), dtype=F32)
    out_dir = torch.zeros((N, 3), dtype=F32)
    out_st = torch.zeros(N, dtype=torch.int8)     # 0 timeout, 1 captured, 2 escaped
    out_lam = torch.zeros(N, dtype=F32)

    camg = cam.to(DEV, F32); fwdg, rightg, upg = (t.to(DEV, F32) for t in (fwd, right, up))
    t0 = time.time()
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk); n = c1 - c0
        s = torch.arange(c0, c1, device=DEV)
        sy = torch.div(s, NW, rounding_mode="floor"); sx = s - sy * NW
        px = (sx.to(F32) + 0.5) / ss; py = (sy.to(F32) + 0.5) / ss
        d = fwdg * f + (px - cx)[:, None] * rightg + (cy - py)[:, None] * upg
        d = d / d.norm(dim=-1, keepdim=True)
        x = camg.expand(n, 3).clone(); v = d.clone()
        lam = -(camg[0] * d[:, 1] - camg[1] * d[:, 0])           # photon L_z / E
        h2 = (torch.linalg.cross(x, v) ** 2).sum(-1, keepdim=True)

        hr = torch.full((K, n), float("nan"), device=DEV, dtype=F32)
        hp = torch.full((K, n), float("nan"), device=DEV, dtype=F32)
        fdir = torch.zeros((n, 3), device=DEV, dtype=F32)
        st = torch.zeros(n, device=DEV, dtype=torch.int8)
        nh = torch.zeros(n, device=DEV, dtype=torch.long)
        idx = torch.arange(n, device=DEV)

        for it in range(max_steps):
            if idx.numel() == 0:
                break
            r = x.norm(dim=-1, keepdim=True)
            dl = (k_step * r).clamp(0.004, 1.5)
            k1x = v;                 k1v = accel(x, h2)
            k2x = v + 0.5 * dl * k1v; k2v = accel(x + 0.5 * dl * k1x, h2)
            k3x = v + 0.5 * dl * k2v; k3v = accel(x + 0.5 * dl * k2x, h2)
            k4x = v + dl * k3v;       k4v = accel(x + dl * k3x, h2)
            xn = x + dl / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            vn = v + dl / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

            z0, z1 = x[:, 2], xn[:, 2]
            cr = (z0 * z1) <= 0
            if cr.any():
                t = (z0 / (z0 - z1 + 1e-30)).clamp(0, 1)
                xc = x + t[:, None] * (xn - x)
                rc = torch.hypot(xc[:, 0], xc[:, 1])
                ok = cr & (rc >= r_in) & (rc <= r_out) & (nh[idx] < K) & (z0 != 0)
                if ok.any():
                    gi = idx[ok]; slot = nh[gi]
                    hr[slot, gi] = rc[ok]
                    hp[slot, gi] = torch.atan2(xc[ok, 1], xc[ok, 0])
                    nh[gi] += 1

            rn = xn.norm(dim=-1)
            cap = rn < 2.0
            esc = (rn > R_esc) & ((xn * vn).sum(-1) > 0)
            done = cap | esc
            if done.any():
                gi = idx[cap]; st[gi] = 1
                gi = idx[esc]; st[gi] = 2
                fdir[gi] = vn[esc] / vn[esc].norm(dim=-1, keepdim=True)
                keep = ~done
                idx, x, v, h2 = idx[keep], xn[keep], vn[keep], h2[keep]
            else:
                x, v = xn, vn
        out_r[:, c0:c1] = hr.cpu(); out_p[:, c0:c1] = hp.cpu()
        out_dir[c0:c1] = fdir.cpu(); out_st[c0:c1] = st.cpu(); out_lam[c0:c1] = lam.cpu()
        if verbose:
            print(f"  chunk {c0//chunk}: {n} rays, {it} steps, left {idx.numel()}, {time.time()-t0:.1f}s", flush=True)
    return dict(r=out_r, p=out_p, dir=out_dir, st=out_st, lam=out_lam, cfg=cfg)
