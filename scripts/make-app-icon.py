#!/usr/bin/env python3
"""Generate the zonos2-app icon set (assets/app/zonos2.{png,ico,icns}).

Draws a 1024x1024 mark in the style of assets/ZONOS2BlogThumbnail.png (near-black
field, orange smoke wave, thin silver "Z2") — the thumbnail itself is 1200x628
wordmark art and doesn't crop to a square. Pillow renders all three outputs, so
this runs on any platform (no iconutil/ImageMagick needed).

    python3 scripts/make-app-icon.py [outdir=assets/app]
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 1024
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/lato/Lato-Light.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "C:/Windows/Fonts/segoeuil.ttf",
]


def load_font(px: int) -> ImageFont.FreeTypeFont:
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            return ImageFont.truetype(cand, px)
    return ImageFont.load_default(px)


def wave_layer(color, amp, phase, thickness, blur, alpha):
    """One blurred sine ribbon across the lower half, like the thumbnail's smoke."""
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pts = []
    for x in range(-64, SIZE + 65, 16):
        y = SIZE * 0.68 + amp * math.sin(x / SIZE * 2.2 * math.pi + phase)
        pts.append((x, y))
    for off in range(-thickness // 2, thickness // 2 + 1, 6):
        d.line([(x, y + off) for x, y in pts], fill=color + (alpha,), width=8)
    return layer.filter(ImageFilter.GaussianBlur(blur))


def make_master() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (10, 11, 14, 255))

    # subtle radial glow behind the mark
    glow = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(glow).ellipse((112, 112, SIZE - 112, SIZE - 112), fill=46)
    img.paste(Image.new("RGBA", (SIZE, SIZE), (46, 42, 36, 255)), (0, 0),
              glow.filter(ImageFilter.GaussianBlur(180)))

    # orange smoke waves (front) + a faint silver one (back)
    img.alpha_composite(wave_layer((190, 190, 200), 120, 2.4, 70, 60, 60))
    img.alpha_composite(wave_layer((222, 130, 28), 96, 0.0, 90, 40, 130))
    img.alpha_composite(wave_layer((255, 170, 60), 110, 0.9, 40, 18, 160))

    # thin silver "Z2", slightly letterspaced, optically centered in the upper field
    font = load_font(430)
    text = "Z2"
    tracking = 40
    widths = [ImageDraw.Draw(img).textlength(c, font=font) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = (SIZE - total) / 2
    top = SIZE * 0.42
    grad = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for i, c in enumerate(text):
        gd.text((x, top), c, font=font, fill=(233, 235, 240, 255), anchor="lm")
        x += widths[i] + tracking
    img.alpha_composite(grad)
    return img


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/app")
    outdir.mkdir(parents=True, exist_ok=True)
    master = make_master()

    master.save(outdir / "zonos2.png")
    master.save(outdir / "zonos2.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    # Pillow's ICNS writer derives all required sizes from the 1024px master.
    master.save(outdir / "zonos2.icns")
    print(f"wrote {outdir}/zonos2.png .ico .icns")


if __name__ == "__main__":
    main()
