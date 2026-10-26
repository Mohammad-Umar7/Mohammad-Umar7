"""Generate assets/awards.svg, the competition-record panel in the profile README.

Run:  python assets/awards.py

To add a result, add a row to TILES. Rank order (1st, 2nd, 3rd, then honours),
newest first within a rank. Tiles sit on a 2-column grid; with an odd count the last
tile is centred. Keep header text under ~350 px (14 px bold caps), measured in Segoe UI.
No entrance animation: first-frame rasterizers would capture blank tiles.
"""
from pathlib import Path

OUT = Path(__file__).with_name("awards.svg")

MEDALS = {
    #          gradient   ring   glyph  glyph-y size  glyph-fill  title-fill
    "gold":   ("mGold",   0.28,  "1",   96,     28,   "#422006",  "#fcd34d"),
    "silver": ("mSilver", 0.3,   "2",   96,     28,   "#1e293b",  "#e2e8f0"),
    "bronze": ("mBronze", 0.28,  "3",   96,     28,   "#431407",  "#fdba74"),
    "steel":  ("mSteel",  0.3,   "★",   94,     23,   "#eff6ff",  "#93c5fd"),
    "violet": ("mViolet", 0.3,   "◆",   94,     23,   "#f5f3ff",  "#c4b5fd"),
}

# (medal, sparkle class or None, header, title, subtitle)
TILES = [
    ("gold",   "spk",      "GAMEFORWARD GAMEJAM: MYTHOS COLLIDE",     "The Twelfth Stamp",           "1st Place · #1 overall · VR horror game for Meta Quest 3"),
    ("gold",   "spk spk2", "QODER AI HACKATHON — AI EVERYTHING AD",   "UnderGrid",                   "1st Place · 6-agent digital twin of underground Abu Dhabi"),
    ("gold",   "spk spk2", "ORIANE × REPLIT HACKATHON — DUBAI",       "PRE//FLIGHT",                 "1st Place · 48 teams · Build for the Video Economy"),
    ("gold",   "spk",      "INSPIRE ’26 — IEEE SSCS AUS CHAPTER",     "AeroHalo",                    "1st Place · University Track · predictive airside safety"),
    ("gold",   "spk",      "TAMKEEN 5.0 — G42 × AD YOUTH COUNCIL",    "Agentic Refund Intelligence", "1st Place · Advanced Track · Agentic AI"),
    ("gold",   "spk spk2", "REPLIT × IEC BUILDATHON — MBZUAI",        "Starkz AI",                   "1st Place · AI safety companion for outdoor crews"),
    ("silver", None,       "EMIRATES AGRICULTURE HACKATHON 2026",     "SooqRoot",                    "2nd Place · awarded by UAE Climate Minister"),
    ("silver", None,       "SAFE AI CUP 2026 — UNIVERSITY OF DUBAI",  "BridgeLab AI",                "2nd Place · bilingual EN/AR STEM lab coach"),
    ("bronze", None,       "C0MPILED-2 HACKATHON — UAE",              "Kanban Team",                 "3rd Place · Yas Marina Circuit · 60+ submissions"),
    ("violet", None,       "AI MICROFILM CHALLENGE — DUBAI",          "The True Architect",          "Excellence Award · 24-hour AI film sprint (Vidu)"),
    ("steel",  None,       "GLOBAL BUILD CHALLENGE — SOCIA",          "Common Ground",               "Finalist · Notch Launch 2026"),
    ("violet", None,       "CREATE APPS CHAMPIONSHIP 2025–26",        "Team Mobile App",             "Recognized · Dubai Chamber of Digital Economy"),
]

TILE_W, TILE_H, GAP, PAD = 480, 160, 20, 10
SPARK = "M0 -6 L1.7 -1.7 L6 0 L1.7 1.7 L0 6 L-1.7 1.7 L-6 0 L-1.7 -1.7 Z"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def tile(x, y, medal, spark, header, title, sub):
    grad, ring, glyph, gy, gsize, gfill, tfill = MEDALS[medal]
    lines = [
        f'  <g transform="translate({x},{y})">',
        f'    <rect width="{TILE_W}" height="{TILE_H}" rx="18" fill="#0c1222" stroke="#1e2a44" stroke-width="1.2"/>',
        f'    <rect x="0" y="0" width="5" height="{TILE_H}" rx="2.5" fill="url(#{grad})"/>',
        '    <path d="M46 22 H58 L55 50 H44 Z" fill="#263250"/>',
        '    <path d="M58 22 H70 L72 50 H61 Z" fill="#1d2b4a"/>',
        f'    <circle cx="58" cy="86" r="34" fill="url(#{grad})" stroke="#0c1222" stroke-width="2"/>',
        f'    <circle cx="58" cy="86" r="25" fill="none" stroke="rgba(255,255,255,{ring})" stroke-width="1.2" stroke-dasharray="2 4"/>',
        f'    <text x="58" y="{gy}" text-anchor="middle" font-size="{gsize}" font-weight="800" fill="{gfill}">{glyph}</text>',
    ]
    if spark:
        lines.append(f'    <path class="{spark}" transform="translate(86,52)" d="{SPARK}" fill="#ffffff"/>')
    lines += [
        f'    <text x="110" y="52" font-size="14" font-weight="700" letter-spacing="0.6" fill="#dbe4f3">{esc(header)}</text>',
        f'    <text x="110" y="84" font-size="19" font-weight="800" fill="{tfill}">{esc(title)}</text>',
        f'    <text x="110" y="112" font-size="13" fill="#8b9bb4">{esc(sub)}</text>',
        '  </g>',
    ]
    return "\n".join(lines)


rows = (len(TILES) + 1) // 2
height = PAD + rows * TILE_H + (rows - 1) * GAP + 2 * PAD
aria = "Competition record: " + ", ".join(f"{t[4].split(' · ')[0]} {t[2].title()} ({t[3]})" for t in TILES)

body = []
for i, t in enumerate(TILES):
    r, c = divmod(i, 2)
    lone = i == len(TILES) - 1 and c == 0
    x = (1000 - TILE_W) // 2 if lone else PAD + c * (TILE_W + GAP)
    body.append(tile(x, PAD + r * (TILE_H + GAP), *t))

svg = f'''<svg width="1000" height="{height}" viewBox="0 0 1000 {height}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(aria)}">
  <title>Competition Record — {len(TILES)} podiums and honors</title>

  <defs>
    <linearGradient id="mGold" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fff7cc"/><stop offset="0.5" stop-color="#fbbf24"/><stop offset="1" stop-color="#92400e"/>
    </linearGradient>
    <linearGradient id="mSilver" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ffffff"/><stop offset="0.5" stop-color="#9fb0c7"/><stop offset="1" stop-color="#3f4c63"/>
    </linearGradient>
    <linearGradient id="mBronze" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ffd9b0"/><stop offset="0.5" stop-color="#ea7c2c"/><stop offset="1" stop-color="#6f2f0f"/>
    </linearGradient>
    <linearGradient id="mSteel" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#dbeafe"/><stop offset="0.5" stop-color="#3b82f6"/><stop offset="1" stop-color="#172554"/>
    </linearGradient>
    <linearGradient id="mViolet" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ede9fe"/><stop offset="0.5" stop-color="#8b5cf6"/><stop offset="1" stop-color="#312e81"/>
    </linearGradient>
  </defs>

  <style>
    text {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif; }}
    @keyframes spark {{ 0%,100% {{ opacity: .15; }} 50% {{ opacity: .95; }} }}
    .spk {{ animation: spark 2.4s ease-in-out infinite; }}
    .spk2 {{ animation-delay: -1.2s; }}
    @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
  </style>

{chr(10).join(body)}
</svg>
'''

OUT.write_text(svg, encoding="utf-8", newline="\n")
print(f"wrote {OUT.name}: {len(TILES)} tiles, {height}px tall")
