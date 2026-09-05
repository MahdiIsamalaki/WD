import math, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

random.seed(21); np.random.seed(21)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "assets", "sprites")
FR = os.path.join(ROOT, "assets", "video", "frames")
os.makedirs(FR, exist_ok=True)

W, H, FPS, DUR = 720, 1280, 30, 6.0
N = int(FPS * DUR)
T_CRACK, T_SPLIT, T_SWEEP0, T_SWEEP_SPREAD, T_LINGER = 1.5, 1.78, 1.95, 1.7, 4.3

paper = Image.open(os.path.join(SP, "paper.png")).convert("RGB").resize((W, H), Image.LANCZOS)
variants = [Image.open(os.path.join(SP, f"petal_{i}.png")).convert("RGBA") for i in range(12)]
seal = Image.open(os.path.join(SP, "seal.png")).convert("RGBA")
seal_ok = Image.open(os.path.join(SP, "seal_ok.png")).convert("RGBA")
seal_sh_src = seal_ok
seal_L = Image.open(os.path.join(SP, "seal_L.png")).convert("RGBA")
seal_R = Image.open(os.path.join(SP, "seal_R.png")).convert("RGBA")
frag = Image.open(os.path.join(SP, "frag.png")).convert("RGBA")

def shadowize(sp, alpha, blur):
    a = sp.split()[3].point(lambda v: int(v * alpha))
    sh = Image.new("RGBA", sp.size, (52, 16, 22, 0))
    sh.putalpha(a.filter(ImageFilter.GaussianBlur(blur)))
    return sh

class Petal:
    def __init__(self, x, y):
        self.x, self.y = x, y
        depth = y / H
        size = int(random.uniform(150, 250) * (1 + 0.5 * depth))
        src = random.choice(variants)
        sp = src.resize((size, int(size * src.height / src.width)), Image.LANCZOS)
        sp = ImageEnhance.Brightness(sp).enhance(random.uniform(0.5, 1.0))
        sp = ImageEnhance.Color(sp).enhance(random.uniform(0.75, 1.15))
        self.a0 = random.uniform(0, 360)
        self.img = sp.rotate(self.a0, expand=True, resample=Image.BICUBIC)
        self.shadow = shadowize(self.img, 0.55, 5).rotate(0, expand=False)
        self.flight = sp  # unrotated copy for in-flight rotation
        self.fshadow = shadowize(self.flight, 0.55, 5)
        self.t0 = None
        self.vx = 0.0; self.vy = 0.0
        self.spin = random.uniform(-170, 170)
        self.swayA = random.uniform(14, 34); self.swayF = random.uniform(3.2, 6.5); self.ph = random.uniform(0, 6.3)
        self.vmax = random.uniform(520, 1050)
        self.gone = False
        self.ox = self.oy = 0
        self.px = self.py = None
    def launch(self, t0):
        self.t0 = t0
        self.vx = random.uniform(90, 240)          # residual drift at lift-off
        self.vy = random.uniform(-260, 120)
    def pos(self, t):
        lt = t - self.t0
        lift = min(1.0, lt / 0.4)
        u = max(0.0, lt - 0.4)
        if u <= 0:
            self.ox = math.sin(self.ph + lt * 7) * 2.2
            self.oy = -16 * math.sin(lift * math.pi / 2) + math.cos(self.ph + lt * 5) * 1.6
        else:
            ramp = 0.9
            if u < ramp:
                dx = 0.5 * (self.vmax / ramp) * u * u + self.vx * u
            else:
                dx = 0.5 * self.vmax * ramp + self.vmax * (u - ramp) + self.vx * ramp
            self.ox = -dx
            self.oy = self.vy * u + 0.5 * 150 * u * u + self.swayA * math.sin(self.swayF * u + self.ph)
        return lift
    def draw(self, fr, t):
        if self.gone:
            return
        if self.t0 is None:
            fr.paste(self.shadow, (int(self.x - self.img.width / 2 + 5), int(self.y - self.img.height / 2 + 9)), self.shadow)
            fr.paste(self.img, (int(self.x - self.img.width / 2), int(self.y - self.img.height / 2)), self.img)
            return
        liftk = self.pos(t)
        ang = self.a0 + self.spin * max(0.0, t - self.t0 - 0.4)
        px = self.x + self.ox; py = self.y + self.oy
        if px < -340 or py > H + 340 or py < -340:
            self.gone = True; return
        sp = self.flight.rotate(ang % 360, expand=True, resample=Image.BILINEAR)
        sh = self.fshadow.rotate(ang % 360, expand=True, resample=Image.BILINEAR)
        sdx = int(5 + 26 * liftk); sdy = int(9 + 34 * liftk)
        fr.paste(sh, (int(px - sp.width / 2 + sdx), int(py - sp.height / 2 + sdy)), sh)
        if self.px is not None and abs(px - self.px) > 16:
            g = Image.new("RGBA", sp.size, (0, 0, 0, 0)); g.paste(sp, (0, 0), sp.split()[3].point(lambda v: int(v * 0.45)))
            fr.paste(g, (int(self.px - sp.width / 2), int(self.py - sp.height / 2)), g)
        fr.paste(sp, (int(px - sp.width / 2), int(py - sp.height / 2)), sp)
        self.px, self.py = px, py

petals = []
row = -60; y = row; band = 0
while y < H + 80:
    step = random.uniform(78, 96)
    x = -80 + (random.uniform(0, 40) if band % 2 else 0)
    while x < W + 80:
        petals.append(Petal(x + random.uniform(-30, 30), y + random.uniform(-34, 34)))
        x += random.uniform(64, 88)
    y += step; band += 1
petals.sort(key=lambda p: p.y)
print("petals:", len(petals))
random.shuffle(petals)
for i, p in enumerate(petals):
    r = random.random()
    p.launch_time = T_SWEEP0 + (r ** 1.35) * T_SWEEP_SPREAD
    if r > 0.93:
        p.launch_time = T_LINGER + random.uniform(0, 0.7)
petals.sort(key=lambda p: p.y)

sx, sy = W / 2, H * 0.47
seal_sh = shadowize(seal_sh_src, 0.65, 10)
dust = []
for i in range(18):
    a = random.uniform(0, 6.28); v = random.uniform(260, 620)
    dust.append({"x": sx + math.cos(a) * 40, "y": sy + math.sin(a) * 40, "vx": math.cos(a) * v, "vy": math.sin(a) * v - 180, "r": random.uniform(0, 360), "s": random.uniform(0.25, 0.8), "t0": T_SPLIT + random.uniform(0, 0.12)})

def ease_out_cubic(x): return 1 - (1 - x) ** 3

vig = Image.new("L", (W, H), 0); dv = ImageDraw.Draw(vig)
dv.ellipse([-W * 0.25, -H * 0.2, W * 1.25, H * 1.2], fill=255)
vig = vig.filter(ImageFilter.GaussianBlur(90))
vign = Image.new("RGBA", (W, H), (28, 8, 12, 255))
vign.putalpha(vig.point(lambda v: int(70 * (1 - v / 255))))

import time
st = time.time()
for f in range(N):
    t = f / FPS
    fr = paper.convert("RGBA")
    for p in petals:
        if not p.gone and p.t0 is None and t > p.launch_time:
            p.launch(p.launch_time)
        p.draw(fr, t)
    # seal
    if t < T_SPLIT:
        sc = 1.0 + 0.015 * math.sin(t * 2.1)
        jit = 0
        if T_CRACK < t < T_SPLIT:
            k = (t - T_CRACK) / (T_SPLIT - T_CRACK)
            jit = math.sin(t * 90) * 3.4 * k
            sc = 1.0
        im = (seal_ok if t < T_CRACK else seal).resize((int(240 * sc), int(240 * sc)))
        sh = seal_sh.resize((int(240 * sc), int(240 * sc)))
        fr.paste(sh, (int(sx - sh.width / 2 + 7), int(sy - sh.height / 2 + 12)), sh)
        fr.paste(im, (int(sx - im.width / 2 + jit), int(sy - im.height / 2 + jit * 0.6)), im)
    else:
        u = min(1.0, (t - T_SPLIT) / 1.05)
        e = ease_out_cubic(u)
        for img, sgn in ((seal_L, -1), (seal_R, 1)):
            im240 = img.resize((240, 240))
            rot = im240.rotate(sgn * 42 * e, expand=True, resample=Image.BILINEAR)
            if u >= 1:
                continue
            a = rot.split()[3].point(lambda v: int(v * (1 - e * e)))
            rot.putalpha(a)
            X = sx + sgn * (16 + 120 * e) - rot.width / 2
            Y = sy + 150 * e * e - rot.height / 2
            fr.paste(rot, (int(X), int(Y)), rot)
        for d in dust:
            u = t - d["t0"]
            if u < 0 or u > 0.9:
                continue
            X = d["x"] + d["vx"] * u * 0.5; Y = d["y"] + d["vy"] * u + 750 * u * u
            al = max(0, 1 - u / 0.9)
            s = int(30 * d["s"])
            g = frag.resize((s, s))
            gg = Image.new("RGBA", g.size, (0, 0, 0, 0)); gg.paste(g, (0, 0), g.split()[3].point(lambda v: int(v * al)))
            fr.paste(gg, (int(X), int(Y)), gg)
    fr = Image.alpha_composite(fr, vign)
    fr.convert("RGB").save(os.path.join(FR, f"f{f:04d}.jpg"), quality=90)
    if f % 30 == 0:
        print(f"frame {f}/{N}  {time.time()-st:.0f}s", flush=True)

# schedule wind launches
print("done", time.time() - st)
