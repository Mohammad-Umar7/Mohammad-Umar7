"""Generate the README's section divider and footer, styled after the black-hole banner.

Run:  python assets/sections.py      (writes assets/divider.svg and assets/footer.svg)

Same palette and type as assets/event-horizon/build.py: amber #FFA64D and cream on deep
space, Syncopate Bold for display text and JetBrains Mono for small caps. Text is converted
to vector paths, because SVGs shown through <img> can't load web fonts.

All motion is SMIL with a visible first frame (no fade-in entrances), so a static
rasterizer still shows the full picture.
"""
import os
import random
import urllib.request

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "event-horizon", ".fonts")  # shared with the banner build (gitignored)
FONTS = {
    "Syncopate-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/apache/syncopate/Syncopate-Bold.ttf",
    "JetBrainsMono[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
}

AMBER, CREAM, WHITE_HOT, MUTED = "#FFA64D", "#F6D4AA", "#FFF6EA", "#B2BCCC"


def font(name, wght=None):
    os.makedirs(FD, exist_ok=True)
    path = os.path.join(FD, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(FONTS[name], path)
    f = TTFont(path)
    if wght is not None and "fvar" in f:
        f = instancer.instantiateVariableFont(f, {"wght": wght})
    return f


def text_path(f, text, size, x, y, tracking=0.0, anchor="start"):
    """SVG path data for `text` set in font `f`, baseline at y."""
    cmap, gs, hmtx = f.getBestCmap(), f.getGlyphSet(), f["hmtx"]
    s = size / f["head"].unitsPerEm
    width = sum(hmtx[cmap.get(ord(c), ".notdef")][0] * s + tracking for c in text) - tracking
    x0 = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
    pen, cx = SVGPathPen(gs, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip(".")), x0
    for c in text:
        g = cmap.get(ord(c), ".notdef")
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, cx, y)))
        cx += hmtx[g][0] * s + tracking
    return pen.getCommands()


def half_ellipse(cx, cy, rx, ry, upper):
    """Left-to-right half ellipse: upper=True passes over the top (the disk's far side)."""
    return f"M{cx - rx:.1f} {cy} A{rx:.1f} {ry:.1f} 0 0 {1 if upper else 0} {cx + rx:.1f} {cy}"


def stars(rng, n, w, h, avoid=None, twinkle=0.3, rmax=1.6, colors=("#ffffff", "#FFE9CC", "#CFE0FF")):
    out = []
    for _ in range(n):
        x, y = rng.uniform(4, w - 4), rng.uniform(4, h - 4)
        if avoid and avoid(x, y):
            continue
        r = rng.uniform(0.5, rmax) ** 1.3
        o = rng.uniform(0.35, 0.9)
        c = rng.choice(colors)
        if rng.random() < twinkle:
            d, b = rng.uniform(2.2, 6.0), rng.uniform(0, 6)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{c}" opacity="{o:.2f}">'
                       f'<animate attributeName="opacity" values="{o:.2f};{o * 0.15:.2f};{o:.2f}" dur="{d:.1f}s" '
                       f'begin="-{b:.1f}s" repeatCount="indefinite"/></circle>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{c}" opacity="{o:.2f}"/>')
    return "\n    ".join(out)


# ------------------------------------------------------------------ divider
def divider():
    W, H, cx, cy = 1600, 72, 800, 36
    particles = []
    for side, (x0, x1) in (("L", (40, cx - 44)), ("R", (W - 66, cx + 18))):
        for i in range(6):
            dur, begin = 4.8, i * 0.8
            particles.append(
                f'<rect x="{x0}" y="{cy - 1}" width="26" height="2" rx="1" fill="url(#pt{side})">'
                f'<animate attributeName="x" values="{x0};{x1}" dur="{dur}s" begin="-{begin}s" repeatCount="indefinite" '
                f'calcMode="spline" keyTimes="0;1" keySplines="0.55 0 0.9 0.6"/>'
                f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.15;0.8;1" dur="{dur}s" '
                f'begin="-{begin}s" repeatCount="indefinite"/></rect>')
    rng = random.Random(11)
    sparkles = []
    for _ in range(10):
        x = rng.choice([rng.uniform(90, 680), rng.uniform(920, 1510)])
        y, s = rng.uniform(12, 60), rng.uniform(2.5, 4.5)
        d, b = rng.uniform(2.4, 4.8), rng.uniform(0, 4)
        sparkles.append(
            f'<path d="M{x:.1f} {y - s:.1f} L{x + s * .28:.1f} {y - s * .28:.1f} L{x + s:.1f} {y:.1f} L{x + s * .28:.1f} {y + s * .28:.1f} '
            f'L{x:.1f} {y + s:.1f} L{x - s * .28:.1f} {y + s * .28:.1f} L{x - s:.1f} {y:.1f} L{x - s * .28:.1f} {y - s * .28:.1f} Z" '
            f'fill="#FFC98A" opacity="0.7"><animate attributeName="opacity" values="0.7;0.08;0.7" dur="{d:.1f}s" '
            f'begin="-{b:.1f}s" repeatCount="indefinite"/></path>')
    back, front = half_ellipse(cx, cy, 54, 6.5, True), half_ellipse(cx, cy, 54, 6.5, False)
    return f'''<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="">
  <defs>
    <linearGradient id="dl" x1="40" y1="0" x2="{W - 40}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{AMBER}" stop-opacity="0"/><stop offset="0.28" stop-color="{AMBER}" stop-opacity="0.55"/>
      <stop offset="0.46" stop-color="{CREAM}"/><stop offset="0.5" stop-color="#ffffff"/><stop offset="0.54" stop-color="{CREAM}"/>
      <stop offset="0.72" stop-color="{AMBER}" stop-opacity="0.55"/><stop offset="1" stop-color="{AMBER}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="ptL" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{AMBER}" stop-opacity="0"/><stop offset="1" stop-color="#FFF1DC"/></linearGradient>
    <linearGradient id="ptR" x1="1" y1="0" x2="0" y2="0"><stop offset="0" stop-color="{AMBER}" stop-opacity="0"/><stop offset="1" stop-color="#FFF1DC"/></linearGradient>
    <radialGradient id="dg" cx="{cx}" cy="{cy}" r="46" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{AMBER}" stop-opacity="0.6"/><stop offset="0.5" stop-color="#FF8A2A" stop-opacity="0.16"/><stop offset="1" stop-color="#FF8A2A" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="dk" x1="{cx - 54}" y1="0" x2="{cx + 54}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#FFF1DC"/><stop offset="0.5" stop-color="{AMBER}"/><stop offset="1" stop-color="#C2621F"/>
    </linearGradient>
  </defs>

  <path d="M40 {cy} H{W - 40}" stroke="url(#dl)" stroke-width="7" stroke-linecap="round" opacity="0.14"/>
  <path d="M40 {cy} H{W - 40}" stroke="url(#dl)" stroke-width="1.6" stroke-linecap="round"/>
  {chr(10).join("  " + s for s in sparkles).strip()}
  {chr(10).join("  " + p for p in particles).strip()}

  <!-- the black hole -->
  <circle cx="{cx}" cy="{cy}" r="46" fill="url(#dg)"/>
  <circle cx="{cx}" cy="{cy}" r="15.5" stroke="{WHITE_HOT}" stroke-width="2.2" opacity="0.9">
    <animate attributeName="opacity" values="0.9;0.55;0.9" dur="3s" repeatCount="indefinite"/>
  </circle>
  <path d="{back}" stroke="url(#dk)" stroke-width="3" opacity="0.9"/>
  <path d="{back}" stroke="#ffffff" stroke-width="1.3" stroke-dasharray="5 9" opacity="0.75">
    <animate attributeName="stroke-dashoffset" values="0;-28" dur="1.1s" repeatCount="indefinite"/>
  </path>
  <circle cx="{cx}" cy="{cy}" r="12.5" fill="#000000"/>
  <path d="{front}" stroke="url(#dk)" stroke-width="3.4"/>
  <path d="{front}" stroke="#ffffff" stroke-width="1.4" stroke-dasharray="5 9" opacity="0.8">
    <animate attributeName="stroke-dashoffset" values="0;28" dur="1.1s" repeatCount="indefinite"/>
  </path>
</svg>
'''


# ------------------------------------------------------------------ footer
def footer():
    W, H, cx, cy = 1200, 300, 600, 206
    mono, mono_b, title = font("JetBrainsMono[wght].ttf", 400), font("JetBrainsMono[wght].ttf", 500), font("Syncopate-Bold.ttf")
    rng = random.Random(42)
    sky = stars(rng, 230, W, H, avoid=lambda x, y: ((x - cx) ** 2 / 9 + (y - cy) ** 2 < 70 ** 2
                                                    or 28 < y < 112 and 250 < x < 950 or y > 262))

    # The disk: a continuous glowing band, then tightly packed rings so it reads as one sheet of gas.
    disk_back, disk_front = [], []
    for layer, upper in ((disk_back, True), (disk_front, False)):
        for rx, w, o in ((260, 40, 0.30), (150, 22, 0.45)):
            layer.append(f'<path d="{half_ellipse(cx, cy, rx, rx * 0.085, upper)}" stroke="url(#disk)" stroke-width="{w}" '
                         f'opacity="{o}" filter="url(#soft)"/>')
        for i, rx in enumerate(range(96, 421, 18)):
            t = (rx - 96) / 324
            w, o = 4.2 - 2.6 * t, 1.0 - 0.8 * t ** 0.8
            layer.append(f'<path d="{half_ellipse(cx, cy, rx, rx * 0.085, upper)}" stroke="url(#disk)" '
                         f'stroke-width="{w:.1f}" opacity="{o:.2f}"/>')
    # swirling streaks: soft dashes travelling over the top left-to-right, under the bottom right-to-left
    for rx, dash, dur in ((132, "26 38", 2.6), (222, "40 70", 3.8), (318, "22 90", 5.2)):
        ry = rx * 0.085
        for layer, upper, sign in ((disk_back, True, -1), (disk_front, False, 1)):
            period = sum(int(v) for v in dash.split())
            layer.append(f'<path d="{half_ellipse(cx, cy, rx, ry, upper)}" stroke="#FFF4E4" stroke-width="1.8" stroke-dasharray="{dash}" '
                         f'stroke-linecap="round" opacity="0.55"><animate attributeName="stroke-dashoffset" '
                         f'values="0;{sign * period}" dur="{dur}s" repeatCount="indefinite"/></path>')

    def orbit(rx):
        ry = rx * 0.085
        return f"M{cx + rx} {cy} A{rx} {ry:.1f} 0 1 1 {cx - rx} {cy} A{rx} {ry:.1f} 0 1 1 {cx + rx} {cy}"

    flares = "".join(
        f'<g><circle r="{r * 2.6:.1f}" fill="{AMBER}" opacity="0.45" filter="url(#soft)"/><circle r="{r}" fill="#FFF8EE"/>'
        f'<animateMotion path="{orbit(rx)}" dur="{dur}s" begin="-{b}s" repeatCount="indefinite"/></g>'
        for rx, r, dur, b in ((270, 4.5, 9, 0), (180, 3.4, 5.6, 2.3), (360, 3.0, 13, 7)))

    over = text_path(mono_b, "THANKS FOR STOPPING BY", 17, cx, 52, tracking=5, anchor="middle")
    head = text_path(title, "LET'S BUILD WHAT'S NEXT", 34, cx, 100, tracking=3, anchor="middle")
    left = text_path(mono, "MOHAMMAD UMAR  ·  ABU DHABI, UAE", 16, 40, 284, tracking=1.2)
    right = text_path(mono, "SEE YOU BEYOND THE EVENT HORIZON", 16, W - 40, 284, tracking=1.2, anchor="end")

    return f'''<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Footer: a black hole with a glowing accretion disk. Thanks for stopping by. Let's build what's next. Mohammad Umar, Abu Dhabi, UAE.">
  <defs>
    <radialGradient id="bgGlow" cx="{cx}" cy="{cy}" r="520" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#3A1C08"/><stop offset="0.35" stop-color="#160A04"/><stop offset="1" stop-color="#03040A" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="disk" x1="{cx - 410}" y1="0" x2="{cx + 410}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{AMBER}" stop-opacity="0"/><stop offset="0.16" stop-color="#FFB566" stop-opacity="0.75"/>
      <stop offset="0.36" stop-color="#FFE6C4"/><stop offset="0.5" stop-color="{WHITE_HOT}"/><stop offset="0.64" stop-color="#FFC27F"/>
      <stop offset="0.84" stop-color="#E0782A" stop-opacity="0.65"/><stop offset="1" stop-color="#A04A12" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="lens" x1="{cx - 92}" y1="0" x2="{cx + 92}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#FFE9CC"/><stop offset="0.5" stop-color="{WHITE_HOT}"/><stop offset="1" stop-color="{AMBER}"/>
    </linearGradient>
    <linearGradient id="name" x1="0" y1="72" x2="0" y2="100" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#FAF6F0"/><stop offset="1" stop-color="{CREAM}"/>
    </linearGradient>
    <linearGradient id="edge" x1="0" y1="0" x2="{W}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{AMBER}" stop-opacity="0"/><stop offset="0.5" stop-color="{AMBER}"/><stop offset="1" stop-color="{AMBER}" stop-opacity="0"/>
    </linearGradient>
    <filter id="soft" x="-50%" y="-200%" width="200%" height="500%"><feGaussianBlur stdDeviation="6"/></filter>
    <clipPath id="farSide"><rect x="0" y="0" width="{W}" height="{cy}"/></clipPath>
    <clipPath id="nearSide"><rect x="0" y="{cy}" width="{W}" height="{H - cy}"/></clipPath>
  </defs>

  <rect width="{W}" height="{H}" fill="#03040A"/>
  <rect width="{W}" height="{H}" fill="url(#bgGlow)"/>
  <g>
    {sky}
  </g>

  <!-- far side: lensed arc over the top, back half of the disk, flares behind the hole -->
  <path d="{half_ellipse(cx, cy, 92, 84, True)}" stroke="url(#lens)" stroke-width="26" opacity="0.45" filter="url(#soft)"/>
  <path d="{half_ellipse(cx, cy, 92, 84, True)}" stroke="url(#lens)" stroke-width="13"/>
  <path d="{half_ellipse(cx, cy, 104, 96, True)}" stroke="{CREAM}" stroke-width="3" opacity="0.45"/>
  {chr(10).join("  " + d for d in disk_back).strip()}
  <g clip-path="url(#farSide)">{flares}</g>

  <!-- the hole -->
  <circle cx="{cx}" cy="{cy}" r="54" stroke="{WHITE_HOT}" stroke-width="6" opacity="0.5" filter="url(#soft)"/>
  <circle cx="{cx}" cy="{cy}" r="54" stroke="{WHITE_HOT}" stroke-width="2.4">
    <animate attributeName="opacity" values="1;0.6;1" dur="3.4s" repeatCount="indefinite"/>
  </circle>
  <circle cx="{cx}" cy="{cy}" r="51" fill="#000000"/>
  <path d="{half_ellipse(cx, cy, 66, 58, False)}" stroke="url(#lens)" stroke-width="4" opacity="0.7"/>

  <!-- near side: front half of the disk crosses in front of the hole -->
  {chr(10).join("  " + d for d in disk_front).strip()}
  <g clip-path="url(#nearSide)">{flares}</g>

  <path d="{over}" fill="{AMBER}"/>
  <path d="{head}" fill="url(#name)"/>
  <path d="{left}" fill="{MUTED}" opacity="0.85"/>
  <path d="{right}" fill="{MUTED}" opacity="0.85"/>
  <rect x="0" y="{H - 3}" width="{W}" height="3" fill="url(#edge)"/>
</svg>
'''


if __name__ == "__main__":
    for name, svg in (("divider.svg", divider()), ("footer.svg", footer())):
        p = os.path.join(HERE, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        print(f"wrote {name}: {len(svg.encode()) / 1024:.1f} KB")
