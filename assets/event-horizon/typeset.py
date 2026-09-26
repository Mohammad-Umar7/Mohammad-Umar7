"""Typography layer for the banner: rendered at 2x, composited in sRGB."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, ".fonts")
S = 2  # supersample factor for text


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FD, name), int(round(size * S)))
    if var is not None:
        try:
            f.set_variation_by_axes(list(var) if isinstance(var, (tuple, list)) else [var])
        except Exception as e:
            print("var fail", name, e)
    return f


def text_width(f, s, track):
    w = 0
    for i, ch in enumerate(s):
        w += f.getlength(ch)
        if i < len(s) - 1:
            w += track * S
    return w / S


def draw_tracked(d, xy, s, f, fill, track):
    x, y = xy[0] * S, xy[1] * S
    for ch in s:
        d.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += f.getlength(ch) + track * S
    return x / S


def layer(W, H):
    return Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))


def gradient_fill(mask, top, bottom, left_tint=None):
    """mask: L image; returns RGBA with a vertical gradient fill and the mask as alpha."""
    w, h = mask.size
    bb = mask.getbbox()
    if bb is None:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y0, y1 = bb[1], bb[3]
    ys = np.clip((np.arange(h) - y0) / max(1, (y1 - y0)), 0, 1)[:, None, None]
    col = np.array(top)[None, None, :] * (1 - ys) + np.array(bottom)[None, None, :] * ys
    col = np.broadcast_to(col, (h, w, 3)).copy()
    if left_tint is not None:
        x0, x1 = bb[0], bb[2]
        xs = np.clip((np.arange(w) - x0) / max(1, (x1 - x0)), 0, 1)[None, :, None]
        col = col * (1 - xs * left_tint[1]) + np.array(left_tint[0])[None, None, :] * xs * left_tint[1]
    a = np.array(mask)[..., None]
    return Image.fromarray(np.concatenate([col, a], -1).astype(np.uint8), "RGBA")


def build_overlay(W, H, spec):
    """returns (rgba text layer, glow mask) at 1x as uint8 arrays."""
    txt = layer(W, H)
    glow_mask = Image.new("L", (W * S, H * S), 0)
    for item in spec:
        f = font(item["font"], item["size"], item.get("var"))
        m = Image.new("L", (W * S, H * S), 0)
        dm = ImageDraw.Draw(m)
        x = item["x"]
        if item.get("align") == "right":
            x = item["x"] - text_width(f, item["text"], item.get("track", 0))
        draw_tracked(dm, (x, item["y"]), item["text"], f, 255, item.get("track", 0))
        if "grad" in item:
            g = gradient_fill(m, item["grad"][0], item["grad"][1], item.get("xtint"))
        else:
            c = item["color"]
            a = np.array(m).astype(np.float32) * item.get("alpha", 1.0)
            g = Image.fromarray(np.dstack([np.full(a.shape, c[0]), np.full(a.shape, c[1]), np.full(a.shape, c[2]), a]).astype(np.uint8), "RGBA")
        txt = Image.alpha_composite(txt, g)
        if item.get("glow", 0) > 0:
            gm = Image.fromarray((np.array(m).astype(np.float32) * item["glow"]).astype(np.uint8))
            glow_mask = Image.fromarray(np.maximum(np.array(glow_mask), np.array(gm)))
    txt = txt.resize((W, H), Image.LANCZOS)
    glow_mask = glow_mask.resize((W, H), Image.LANCZOS)
    return np.array(txt), np.array(glow_mask)
