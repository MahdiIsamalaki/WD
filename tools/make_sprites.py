import math, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(7); np.random.seed(7)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "sprites")
os.makedirs(OUT, exist_ok=True)

rose = Image.open(os.path.join(ROOT, "assets", "rose_c2.jpg")).convert("RGB")

def smooth_noise(n, octaves=3):
    v = np.zeros(n)
    for o in range(octaves):
        k = 2 ** o
        pts = np.random.randn(k + 2)
        xs = np.linspace(0, k + 1, n, endpoint=False)
        idx = xs.astype(int); fr = xs - idx
        fr = fr * fr * (3 - 2 * fr)
        v += (pts[idx] * (1 - fr) + pts[idx + 1] * fr) / (2 ** o)
    return v

def petal_mask(S, ruffle=0.05):
    ss = 4; A = Image.new("L", (S * ss, S * ss), 0)
    d = ImageDraw.Draw(A); c = S * ss / 2
    n = 180; th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rx, ry = 0.46, 0.56
    r = 1 / np.sqrt((np.cos(th) / rx) ** 2 + (np.sin(th) / ry) ** 2)
    r *= 1 - 0.30 * np.exp(-((th - math.pi / 2) / 0.55) ** 2)                 # narrow base at bottom
    topd = np.minimum(np.abs(th - 1.5 * math.pi), 2 * math.pi - np.abs(th - 1.5 * math.pi))
    r *= 1 - 0.13 * np.exp(-(topd ** 2) * 90)                                 # center notch at top
    r *= 1 + 0.05 * np.exp(-((topd - 0.5) ** 2) * 20)                         # two lobes
    r *= 1 + ruffle * smooth_noise(n, 3) * 0.6
    pts = [(c + r[i] * c * math.cos(th[i]), c + r[i] * c * math.sin(th[i])) for i in range(n)]
    d.polygon(pts, fill=255)
    A = A.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.6))
    return A

def best_patch(PW, PH, tries=400):
    W, H = rose.size; arr = np.asarray(rose).astype(np.int16)
    best, best_s = (500, 400), -1e9
    for _ in range(tries):
        x = random.randint(0, W - PW - 1); y = random.randint(0, H - PH - 1)
        p = arr[y:y + PH, x:x + PW]
        r, g, b = p[..., 0].mean(), p[..., 1].mean(), p[..., 2].mean()
        lum = (r + g + b) / 3
        if not (60 < lum < 170):
            continue
        score = (r - max(g, b)) * 1.5 + lum * 0.25 - abs(lum - 112) * 0.8
        if score > best_s:
            best_s = score; best = (x, y)
    return rose.crop((best[0], best[1], best[0] + PW, best[1] + PH))

def make_petal(idx, S):
    patch = best_patch(S, S).resize((S, S), Image.LANCZOS).convert("RGBA")
    a = np.asarray(patch).astype(np.float32)
    yy = np.mgrid[0:S, 0:S][1] / S
    xx = np.mgrid[0:S, 0:S][0] / S
    shade = 0.68 + 0.55 * (1 - yy) + 0.12 * (xx - 0.5)
    shade = np.clip(shade, 0.5, 1.2) * random.uniform(0.8, 1.05)
    a = np.clip(a * shade[..., None], 0, 255)
    a[..., 3] = np.asarray(petal_mask(S, ruffle=random.uniform(0.035, 0.07))).astype(np.float32)
    Image.fromarray(a.astype(np.uint8), "RGBA").save(os.path.join(OUT, f"petal_{idx}.png"))

for i in range(12):
    make_petal(i, [300, 320, 340, 360][i % 4])

def blob_shape(S, rad=0.42, seed=5):
    rng = np.random.RandomState(seed)
    ss = 4; A = Image.new("L", (S * ss, S * ss), 0); d = ImageDraw.Draw(A); c = S * ss / 2
    n = 260; th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    noise = np.zeros(n)
    for o in range(3):
        k = 2 ** o; p = rng.randn(k + 2)
        xs = np.linspace(0, k + 1, n, endpoint=False); i0 = xs.astype(int); fr = xs - i0
        fr = fr * fr * (3 - 2 * fr)
        noise += (p[i0] * (1 - fr) + p[i0 + 1] * fr) / (2 ** o)
    r = rad * (1 + 0.045 * noise)
    r += 0.06 * rad * np.exp(-((th - 5.5) ** 2) * 40)
    r += 0.05 * rad * np.exp(-((th - 2.4) ** 2) * 60)
    pts = [(c + r[i] * 2 * c * math.cos(th[i]), c + r[i] * 2 * c * math.sin(th[i])) for i in range(n)]
    d.polygon(pts, fill=255)
    return A.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.1))

def heart_pts(cx, cy, sc):
    t = np.linspace(0, 2 * math.pi, 120)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return [(cx + xv * sc, cy + yv * sc - 2 * sc) for xv, yv in zip(x, y)]

def make_seal():
    S = 520
    sh = blob_shape(S, 0.40, seed=11)
    yy, xx = np.mgrid[0:S, 0:S]
    d = np.sqrt((xx - S * 0.44) ** 2 + (yy - S * 0.38) ** 2) / (S * 0.44)
    c0, c1, c2, c3 = np.array([212, 124, 132]), np.array([166, 62, 72]), np.array([112, 32, 43]), np.array([76, 17, 28])
    t = np.clip(d, 0, 1)[..., None]
    col = c0 + (c1 - c0) * np.clip(t / 0.35, 0, 1)
    col = np.where(t < 0.35, col, c1 + (c2 - c1) * np.clip((t - 0.35) / 0.4, 0, 1))
    col = np.where(t < 0.75, col, c2 + (c3 - c2) * np.clip((t - 0.75) / 0.25, 0, 1))
    a = np.zeros((S, S, 4), np.float32); a[..., :3] = col
    a[..., :3] += (np.random.RandomState(2).randn(S, S, 1) * 5)
    a[..., 3] = np.asarray(sh)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(img, "RGBA")
    cx, cy, R = S / 2, S / 2, S * 0.29
    for a0 in range(0, 360, 16):
        dr.arc([cx - R, cy - R, cx + R, cy + R], a0, a0 + 8, fill=(246, 214, 196, 95), width=5)
    hp = heart_pts(cx, cy, S * 0.0105)
    dr.polygon([(x + 3, y + 4) for x, y in hp], fill=(232, 186, 168, 120))
    dr.polygon(hp, fill=(78, 16, 28, 235))
    gloss = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(gloss).ellipse([S * 0.13, S * 0.05, S * 0.60, S * 0.40], fill=(255, 238, 232, 52))
    gloss = gloss.filter(ImageFilter.GaussianBlur(30))
    inner = Image.new("RGBA", (S, S), (0, 0, 0, 0)); dr2 = ImageDraw.Draw(inner)
    sh_dilate = sh.filter(ImageFilter.MinFilter(5))
    edge = Image.new("L", (S, S), 0)
    dr3 = ImageDraw.Draw(edge); dr3.bitmap((0, 0), sh_dilate, fill=255)
    img = Image.alpha_composite(img, Image.composite(gloss, Image.new("RGBA", (S, S), (0, 0, 0, 0)), sh))
    img.save(os.path.join(OUT, "seal.png"))

    rng = random.Random(9)
    img.save(os.path.join(OUT, "seal_ok.png"))
    path = [(S * 0.5 + rng.uniform(-14, 14), -6)]
    for yv in np.linspace(0, S, 9)[1:-1]:
        path.append((S * 0.5 + rng.uniform(-30, 30), yv))
    path.append((S * 0.5 + rng.uniform(-14, 14), S + 6))
    cutL = Image.new("L", (S, S), 0); ImageDraw.Draw(cutL).polygon(path + [(S + 8, S), (S + 8, -8)], fill=255)
    cutR = Image.new("L", (S, S), 0); ImageDraw.Draw(cutR).polygon(path + [(-8, S), (-8, -8)], fill=255)
    final = img.copy(); ImageDraw.Draw(final, "RGBA").line(path, fill=(48, 10, 16, 235), width=3)
    final.save(os.path.join(OUT, "seal.png"))
    def masked(maskL):
        out = final.copy()
        alpha = Image.new("L", (S, S), 0)
        alpha.paste(Image.composite(sh, Image.new("L", (S, S), 0), maskL), (0, 0))
        out.putalpha(alpha)
        return out
    masked(cutL).save(os.path.join(OUT, "seal_L.png"))
    masked(cutR).save(os.path.join(OUT, "seal_R.png"))

make_seal()

def make_paper():
    Wp, Hp = 760, 1352
    yy, xx = np.mgrid[0:Hp, 0:Wp]
    top, bot = np.array([248, 243, 233]), np.array([241, 232, 218])
    t = (yy / Hp)[..., None]
    base = top * (1 - t) + bot * t
    base += np.random.randn(Hp, Wp, 1).astype(np.float32) * 1.5
    fx = (xx / Wp - 0.5) ** 2 + (yy / Hp - 0.5) ** 2
    base -= (fx * 40)[..., None] * np.array([0.55, 0.42, 0.30])[None, None, :]
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "paper.png"))

make_paper()

def make_frag():
    S = 46; yy, xx = np.mgrid[0:S, 0:S]
    d = np.sqrt((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / (S / 2)
    a = np.zeros((S, S, 4), np.uint8); a[..., 0] = 150; a[..., 1] = 52; a[..., 2] = 62
    a[..., 3] = np.clip((1 - d) * 300, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(os.path.join(OUT, "frag.png"))

make_frag()
print("sprites done:", sorted(os.listdir(OUT)))
