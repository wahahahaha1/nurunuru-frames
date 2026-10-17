"""Generates the extension icons (stdlib only): a rounded blue tile with a play triangle and motion trail."""
import os, struct, zlib

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extension", "icons")


def png(path, w, h, rgba):
    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4:(y + 1) * w * 4]) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(data)


def inside_tri(x, y, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    d1 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
    d2 = (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)
    d3 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def inside_rrect(x, y, r):
    cx = min(max(x, r), 1 - r)
    cy = min(max(y, r), 1 - r)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def render(size, ss=4):
    out = []
    tris = [(-0.20, 0.30), (-0.10, 0.55), (0.0, 1.0)]  # x shift, opacity
    for py in range(size):
        for px in range(size):
            acc = [0.0, 0.0, 0.0, 0.0]
            for sy in range(ss):
                for sx in range(ss):
                    x = (px + (sx + 0.5) / ss) / size
                    y = (py + (sy + 0.5) / ss) / size
                    if not inside_rrect(x, y, 0.22):
                        continue
                    t = y
                    col = [0.10 + 0.05 * t, 0.42 - 0.12 * t, 0.95 - 0.25 * t]
                    for shift, op in tris:
                        tri = ((0.36 + shift, 0.26), (0.36 + shift, 0.74), (0.76 + shift, 0.50))
                        if inside_tri(x, y, tri):
                            col = [c * (1 - op) + op for c in col]
                    acc[0] += col[0]
                    acc[1] += col[1]
                    acc[2] += col[2]
                    acc[3] += 1
            n = ss * ss
            a = acc[3] / n
            if acc[3]:
                out += [round(255 * acc[0] / acc[3]), round(255 * acc[1] / acc[3]), round(255 * acc[2] / acc[3]), round(255 * a)]
            else:
                out += [0, 0, 0, 0]
    return out


os.makedirs(OUT, exist_ok=True)
for s in (16, 32, 48, 128):
    png(os.path.join(OUT, f"icon{s}.png"), s, s, render(s))
    print("icon", s)
