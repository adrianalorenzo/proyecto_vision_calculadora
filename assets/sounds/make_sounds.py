# Synthesizes all the sound effects and the 8-bit background music of AirMath with numpy.
# Run it once from this folder: python make_sounds.py
import numpy as np
import wave, os

sr = 44100
out = os.path.dirname(os.path.abspath(__file__))     # this folder
os.makedirs(out, exist_ok=True)

def note(n):  # MIDI note -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)

def t_(d): return np.arange(int(sr * d)) / sr

def sine(f, d): return np.sin(2 * np.pi * f * t_(d))
def square(f, d, duty=0.5):
    ph = (f * t_(d)) % 1.0
    return np.where(ph < duty, 1.0, -1.0)
def tri(f, d):
    ph = (f * t_(d)) % 1.0
    return 4 * np.abs(ph - 0.5) - 1
def noise(d, seed=0): return np.random.default_rng(seed).uniform(-1, 1, int(sr * d))

def env(x, a=0.005, r=None, decay=None):
    n = len(x); e = np.ones(n)
    na = max(1, int(a * sr)); e[:na] = np.linspace(0, 1, na)
    if decay is not None:
        e *= np.exp(-np.arange(n) / sr / decay)
    if r is not None:
        nr = min(n, int(r * sr)); e[-nr:] *= np.linspace(1, 0, nr)
    return x * e

def lowpass(x, k=6):
    return np.convolve(x, np.ones(k) / k, mode="same")

def cat(*xs): return np.concatenate(xs)
def silence(d): return np.zeros(int(sr * d))

def mix_at(base, x, start):
    i = int(start * sr); end = min(len(base), i + len(x)); base[i:end] += x[:end - i]; return base

def save(name, x, gain=0.8):
    x = x / (np.max(np.abs(x)) + 1e-9) * gain
    data = (x * 32767).astype(np.int16)
    with wave.open(os.path.join(out, name + ".wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(data.tobytes())

# --- effects ---
save("click", env(sine(1800, 0.05) + 0.3 * sine(3600, 0.05), 0.002, decay=0.012), 0.5)
save("card_ok", lowpass(cat(env(square(note(81), 0.07, 0.25), 0.003, 0.02), env(square(note(88), 0.11, 0.25), 0.003, 0.06))), 0.55)
save("error", lowpass(cat(env(square(note(52), 0.12), 0.003, 0.03), env(square(note(47), 0.22), 0.003, 0.1)), 10), 0.6)
save("locked", lowpass(env(square(110, 0.6) * (0.6 + 0.4 * np.sin(2 * np.pi * 12 * t_(0.6))), 0.005, 0.2), 12), 0.6)
seq = [72, 76, 79, 84, 88]
x = cat(*[env(square(note(n), 0.075, 0.25) + 0.5 * tri(note(n), 0.075), 0.003, 0.02) for n in seq])
chord = sum(env(tri(note(n), 0.45) + 0.3 * square(note(n), 0.45, 0.25), 0.005, decay=0.25) for n in [72, 76, 79, 84])
save("unlock", lowpass(cat(x, chord), 5), 0.7)
bell = lambda n, d: env(sine(note(n), d) + 0.35 * sine(note(n) * 2, d) + 0.15 * sine(note(n) * 3.01, d), 0.002, decay=0.18)
c = np.zeros(int(sr * 0.6)); mix_at(c, bell(88, 0.5), 0); mix_at(c, bell(95, 0.5), 0.08)
save("correct", c, 0.6)
p = np.zeros(int(sr * 0.7))
for i, n in enumerate([79, 84, 88, 91]): mix_at(p, env(square(note(n), 0.09, 0.25) + tri(note(n), 0.09), 0.003, 0.03), 0.06 * i)
mix_at(p, bell(96, 0.4) * 0.5, 0.26)
save("point", lowpass(p, 4), 0.65)
f = np.zeros(int(sr * 1.4))
for i, n in enumerate([72, 76, 79]): mix_at(f, env(square(note(n), 0.14, 0.25) + tri(note(n), 0.14), 0.003, 0.04), 0.15 * i)
fin = sum(env(tri(note(n), 0.9) + 0.4 * square(note(n), 0.9, 0.25), 0.005, decay=0.45) for n in [72, 76, 79, 84])
mix_at(f, fin, 0.45)
save("game_over", lowpass(f, 5), 0.7)
r = f.copy()
for i, n in enumerate([96, 100, 103, 108]): mix_at(r, bell(n, 0.4) * 0.4, 0.5 + 0.07 * i)
save("record", lowpass(r, 4), 0.75)
sw = noise(0.25, 3) * np.linspace(1, 0, int(sr * 0.25)) ** 2
save("reset", lowpass(sw, 20), 0.35)

# --- background music: 8-bit loop, Am F C G, 112 bpm, 8 bars ---
bpm = 112; beat = 60 / bpm; bar = 4 * beat; bars = 8
L = int(sr * bar * bars); music = np.zeros(L)
prog = [(57, [69, 72, 76]), (53, [65, 69, 72]), (48, [64, 67, 72]), (55, [67, 71, 74])] * 2
s16 = beat / 4
for b, (root, ch) in enumerate(prog):
    t0 = b * bar
    for k in range(8):   # bass eighths (triangle)
        n = root if k % 2 == 0 else root + 12
        mix_at(music, 0.55 * env(tri(note(n - 12), beat / 2 * 0.9), 0.004, 0.03), t0 + k * beat / 2)
    arp = ch + [ch[1] + 12]
    for k in range(16):  # arpeggio sixteenths (soft square)
        n = arp[k % 4] + (12 if (b % 2 == 1 and k >= 8) else 0)
        mix_at(music, 0.16 * lowpass(env(square(note(n), s16 * 0.8, 0.25), 0.002, 0.02, decay=0.08), 8), t0 + k * s16)
    for k in range(4):   # kick on every beat, hat on off-beats
        kt = t_(0.12); kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 25)
        mix_at(music, 0.5 * kick, t0 + k * beat)
        mix_at(music, 0.07 * env(noise(0.03, b * 4 + k), 0.001, decay=0.01), t0 + k * beat + beat / 2)
music = lowpass(music, 3)
save("music", music, 0.5)
print(sorted(os.listdir(out)), round(L / sr, 2), "s")
