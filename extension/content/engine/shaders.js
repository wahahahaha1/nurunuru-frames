// WGSL sources for the NuruNuru Frames interpolation engine.
// Conventions:
//   * Luma pyramids live in plain f32 storage buffers (one buffer per level).
//   * Forward field F: blocks on frame A, vector v with A(x) ~ B(x + v).
//     Backward field G: blocks on frame B, vector w with B(x) ~ A(x + w).
//   * Fields handed to the renderer are rgba16float textures (one texel per block)
//     holding (v.x, v.y) in UV units, z = matching SAD per pixel (0..1 luma scale).
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  const fullscreenVS = /* wgsl */ `
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
  return o;
}
`;

  // Video frame (texture_external) -> rgba8unorm frame texture, with a 2x2 tap box filter when downscaling.
  const ingest = fullscreenVS + /* wgsl */ `
struct IngestU { texel: vec2f, taps: f32, pad: f32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var ext: texture_external;
@group(0) @binding(2) var<uniform> U: IngestU;
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  if (U.taps < 1.5) {
    return vec4f(textureSampleBaseClampToEdge(ext, samp, in.uv).rgb, 1.0);
  }
  let d = U.texel * 0.25;
  let c = textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f(-d.x, -d.y)).rgb
        + textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f( d.x, -d.y)).rgb
        + textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f(-d.x,  d.y)).rgb
        + textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f( d.x,  d.y)).rgb;
  return vec4f(c * 0.25, 1.0);
}
`;

  // Frame texture -> luma at motion-estimation resolution (box filter of taps x taps bilinear samples).
  const luma = /* wgsl */ `
struct LU { w: u32, h: u32, taps: u32, pad: u32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outL: array<f32>;
@group(0) @binding(3) var<uniform> U: LU;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.w || g.y >= U.h) { return; }
  let inv = vec2f(1.0 / f32(U.w), 1.0 / f32(U.h));
  let base = vec2f(g.xy) * inv;
  let n = U.taps;
  var s = 0.0;
  for (var j = 0u; j < n; j++) {
    for (var i = 0u; i < n; i++) {
      let uv = base + (vec2f(f32(i), f32(j)) + 0.5) / f32(n) * inv;
      s += dot(textureSampleLevel(img, samp, uv, 0.0).rgb, vec3f(0.299, 0.587, 0.114));
    }
  }
  outL[g.y * U.w + g.x] = s / f32(n * n);
}
`;

  const downsample = /* wgsl */ `
struct DU { iw: u32, ih: u32, ow: u32, oh: u32 };
@group(0) @binding(0) var<storage, read> inL: array<f32>;
@group(0) @binding(1) var<storage, read_write> outL: array<f32>;
@group(0) @binding(2) var<uniform> U: DU;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.ow || g.y >= U.oh) { return; }
  let x0 = min(g.x * 2u, U.iw - 1u);
  let x1 = min(g.x * 2u + 1u, U.iw - 1u);
  let y0 = min(g.y * 2u, U.ih - 1u);
  let y1 = min(g.y * 2u + 1u, U.ih - 1u);
  outL[g.y * U.ow + g.x] = 0.25 * (inL[y0 * U.iw + x0] + inL[y0 * U.iw + x1] + inL[y1 * U.iw + x0] + inL[y1 * U.iw + x1]);
}
`;

  // Hierarchical block matching, one workgroup per block.
  //  top level  (useParent = 0): exhaustive search within +-radius (+ temporal candidate)
  //  lower levels (useParent = 1): parent 3x3 neighbourhood (x2), temporal, zero -> best,
  //                                then full search within +-radius around it
  //  subpel > 0: that many rounds of 8-neighbour sub-pixel refinement (1/2, 1/4 ...)
  const motion = /* wgsl */ `
struct MEP {
  w: u32, h: u32, nbx: u32, nby: u32,
  pnbx: u32, pnby: u32, bs: u32, ov: u32,
  radius: i32, useParent: u32, useTemporal: u32, subpel: u32,
  lambda: f32, lambdaZero: f32, pad0: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> P: MEP;
@group(0) @binding(1) var<storage, read> cur: array<f32>;
@group(0) @binding(2) var<storage, read> refL: array<f32>;
@group(0) @binding(3) var<storage, read> parent: array<vec4f>;
@group(0) @binding(4) var<storage, read> temporal: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outv: array<vec4f>;

var<workgroup> blk: array<f32, 1024>;
var<workgroup> win: array<f32, 1600>;
var<workgroup> ps: array<f32, 64>;
var<workgroup> rc: array<f32, 64>;
var<workgroup> rv: array<vec2f, 64>;
var<workgroup> pred: vec2f;

// Each candidate's SAD is split over a group of 4 lanes (lane l sums rows l, l+4, ...);
// the 64 threads therefore evaluate 16 candidates at a time.

fn refAt(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(P.w) - 1);
  let cy = clamp(y, 0, i32(P.h) - 1);
  return refL[u32(cy) * P.w + u32(cx)];
}

fn sadGlobalPart(v: vec2i, x0: i32, y0: i32, ws: i32, lane: i32) -> f32 {
  var s = 0.0;
  for (var j = lane; j < ws; j += 4) {
    for (var i = 0; i < ws; i++) {
      s += abs(blk[j * ws + i] - refAt(x0 + i + v.x, y0 + j + v.y));
    }
  }
  return s;
}

// d = position (window pixels) of the block's top-left inside the shared search window.
fn sadWinPart(d: vec2i, ws: i32, side: i32, lane: i32) -> f32 {
  var s = 0.0;
  for (var j = lane; j < ws; j += 4) {
    let row = (j + d.y) * side + d.x;
    for (var i = 0; i < ws; i++) {
      s += abs(blk[j * ws + i] - win[row + i]);
    }
  }
  return s;
}

fn sadWinSubPart(d: vec2f, ws: i32, side: i32, lane: i32) -> f32 {
  let fd = floor(d);
  let a = d - fd;
  let o = vec2i(fd);
  var s = 0.0;
  for (var j = lane; j < ws; j += 4) {
    let r0 = (j + o.y) * side + o.x;
    let r1 = r0 + side;
    for (var i = 0; i < ws; i++) {
      let top = mix(win[r0 + i], win[r0 + i + 1], a.x);
      let bot = mix(win[r1 + i], win[r1 + i + 1], a.x);
      s += abs(blk[j * ws + i] - mix(top, bot, a.y));
    }
  }
  return s;
}

// Sum of the 4 partial SADs of this thread's group (call from uniform control flow).
fn groupSum(tid: u32, val: f32) -> f32 {
  ps[tid] = val;
  workgroupBarrier();
  let b = tid & 0xFFFFFFFCu;
  let s = ps[b] + ps[b + 1u] + ps[b + 2u] + ps[b + 3u];
  workgroupBarrier();
  return s;
}

fn cost(v: vec2f, sad: f32, npix: f32) -> f32 {
  let d = abs(v - pred);
  return sad / npix + P.lambda * (d.x + d.y) + P.lambdaZero * (abs(v.x) + abs(v.y));
}

fn reduceBest(tid: u32) {
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s && rc[tid + s] < rc[tid]) {
      rc[tid] = rc[tid + s];
      rv[tid] = rv[tid + s];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let bx = wg.x;
  let by = wg.y;
  let lane = i32(tid & 3u);
  let cand = i32(tid >> 2u);
  let ws = i32(P.bs + 2u * P.ov);
  let x0 = i32(bx * P.bs) - i32(P.ov);
  let y0 = i32(by * P.bs) - i32(P.ov);
  for (var k = tid; k < u32(ws * ws); k += 64u) {
    let cx = clamp(x0 + i32(k) % ws, 0, i32(P.w) - 1);
    let cy = clamp(y0 + i32(k) / ws, 0, i32(P.h) - 1);
    blk[k] = cur[u32(cy) * P.w + u32(cx)];
  }
  if (tid == 0u) {
    var pr = vec2f(0.0);
    if (P.useParent == 1u) {
      let pbx = min(bx / 2u, P.pnbx - 1u);
      let pby = min(by / 2u, P.pnby - 1u);
      pr = parent[pby * P.pnbx + pbx].xy * 2.0;
    } else if (P.useTemporal == 1u) {
      pr = temporal[by * P.nbx + bx].xy;
    }
    pred = pr;
  }
  workgroupBarrier();
  let npix = f32(ws * ws);
  var bestC = 1e30;
  var bestV = vec2f(0.0);

  // Stage 1: exhaustive search (top level) or predictor candidates (lower levels).
  if (P.useParent == 0u) {
    let r = P.radius;
    let n = 2 * r + 1;
    let total = n * n + 1; // + temporal predictor
    for (var base = 0; base < total; base += 16) {
      let idx = base + cand;
      var valid = idx < total;
      var v = round(pred);
      if (idx < n * n) {
        v = vec2f(f32(idx % n - r), f32(idx / n - r));
      } else {
        valid = valid && P.useTemporal == 1u;
      }
      var part = 0.0;
      if (valid) { part = sadGlobalPart(vec2i(v), x0, y0, ws, lane); }
      let sad = groupSum(tid, part);
      if (valid && lane == 0) {
        let c = cost(v, sad, npix);
        if (c < bestC) { bestC = c; bestV = v; }
      }
    }
  } else {
    // 9 parent neighbourhood vectors (x2), temporal, zero
    var v = vec2f(0.0);
    var valid = cand < 11;
    if (cand < 9) {
      let qx = clamp(i32(min(bx / 2u, P.pnbx - 1u)) + cand % 3 - 1, 0, i32(P.pnbx) - 1);
      let qy = clamp(i32(min(by / 2u, P.pnby - 1u)) + cand / 3 - 1, 0, i32(P.pnby) - 1);
      v = parent[u32(qy) * P.pnbx + u32(qx)].xy * 2.0;
    } else if (cand == 9) {
      valid = P.useTemporal == 1u;
      v = temporal[by * P.nbx + bx].xy;
    }
    v = round(v);
    var part = 0.0;
    if (valid) { part = sadGlobalPart(vec2i(v), x0, y0, ws, lane); }
    let sad = groupSum(tid, part);
    if (valid && lane == 0) {
      bestC = cost(v, sad, npix);
      bestV = v;
    }
  }
  rc[tid] = bestC;
  rv[tid] = bestV;
  reduceBest(tid);

  // Stage 2: load a search window around the best integer vector into shared memory.
  let r = select(0, P.radius, P.useParent == 1u);
  let m = r + 1;
  let side = ws + 2 * m;
  let c0 = rv[0];
  let cc0 = rc[0];
  workgroupBarrier();
  if (r > 0 || P.subpel > 0u) {
    let ox = x0 + i32(c0.x) - m;
    let oy = y0 + i32(c0.y) - m;
    for (var k = i32(tid); k < side * side; k += 64) {
      win[k] = refAt(ox + k % side, oy + k / side);
    }
    workgroupBarrier();
  }

  // Stage 3: full search within +-r around c0.
  if (r > 0) {
    let n = 2 * r + 1;
    bestC = 1e30;
    bestV = c0;
    if (tid == 0u) { bestC = cc0; }
    for (var base = 0; base < n * n; base += 16) {
      let idx = base + cand;
      let valid = idx < n * n;
      let d = vec2i(idx % n - r, idx / n - r);
      var part = 0.0;
      if (valid) { part = sadWinPart(d + vec2i(m), ws, side, lane); }
      let sad = groupSum(tid, part);
      if (valid && lane == 0) {
        let v = c0 + vec2f(d);
        let c = cost(v, sad, npix);
        if (c < bestC) { bestC = c; bestV = v; }
      }
    }
    rc[tid] = bestC;
    rv[tid] = bestV;
    reduceBest(tid);
  }

  // Stage 4: sub-pixel refinement (bilinear) inside the window, 8 neighbours per round.
  var step = 0.5;
  for (var it = 0u; it < P.subpel; it++) {
    let cb = rv[0];
    let ccb = rc[0];
    workgroupBarrier();
    bestC = 1e30;
    bestV = cb;
    if (tid == 0u) { bestC = ccb; }
    let valid = cand < 8;
    let k = select(cand, cand + 1, cand >= 4);
    let v = cb + vec2f(f32(k % 3 - 1), f32(k / 3 - 1)) * step;
    let d = clamp(v - c0 + f32(m), vec2f(0.0), vec2f(f32(2 * m)));
    var part = 0.0;
    if (valid) { part = sadWinSubPart(d, ws, side, lane); }
    let sad = groupSum(tid, part);
    if (valid && lane == 0) {
      let c = cost(v, sad, npix);
      if (c < bestC) { bestC = c; bestV = v; }
    }
    rc[tid] = bestC;
    rv[tid] = bestV;
    reduceBest(tid);
    step = step * 0.5;
  }

  // Final: plain SAD of the chosen vector (group 0) for reliability / scene detection.
  let vf = rv[0];
  var part = 0.0;
  if (cand == 0) {
    if (r > 0 || P.subpel > 0u) {
      part = sadWinSubPart(clamp(vf - c0 + f32(m), vec2f(0.0), vec2f(f32(2 * m))), ws, side, lane);
    } else {
      part = sadGlobalPart(vec2i(vf), x0, y0, ws, lane);
    }
  }
  let sad = groupSum(tid, part);
  if (tid == 0u) {
    outv[by * P.nbx + bx] = vec4f(vf, sad / npix, 0.0);
  }
}
`;

  // 3x3 component-wise vector median -> rgba16float field texture (UV units) + scene-change statistics.
  const finalize = /* wgsl */ `
struct FU { nbx: u32, nby: u32, w: u32, h: u32, sceneSad: f32, useMedian: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> vin: array<vec4f>;
@group(0) @binding(1) var fieldOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 4>;
@group(0) @binding(3) var<uniform> U: FU;

fn med9(a: array<f32, 9>) -> f32 {
  var v = a;
  for (var i = 1; i < 9; i++) {
    let x = v[i];
    var j = i - 1;
    while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
    v[j + 1] = x;
  }
  return v[4];
}

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.nbx || g.y >= U.nby) { return; }
  let c = vin[g.y * U.nbx + g.x];
  var v = c.xy;
  if (U.useMedian == 1u) {
    var xs: array<f32, 9>;
    var ys: array<f32, 9>;
    var n = 0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let qx = clamp(i32(g.x) + dx, 0, i32(U.nbx) - 1);
        let qy = clamp(i32(g.y) + dy, 0, i32(U.nby) - 1);
        let q = vin[u32(qy) * U.nbx + u32(qx)];
        xs[n] = q.x;
        ys[n] = q.y;
        n++;
      }
    }
    v = vec2f(med9(xs), med9(ys));
  }
  textureStore(fieldOut, g.xy, vec4f(v.x / f32(U.w), v.y / f32(U.h), c.z, 0.0));
  atomicAdd(&stats[1], 1u);
  if (c.z > U.sceneSad) { atomicAdd(&stats[0], 1u); }
}
`;

  // Motion-compensated interpolation, rendered straight into the output (canvas or test texture).
  // mode: 0 interpolate, 1 frame A, 2 frame B, 3 plain blend, 4 vector visualisation
  const render = fullscreenVS + /* wgsl */ `
struct RU {
  rect: vec4f,
  fieldScale: vec2f,
  t: f32,
  mode: u32,
  masking: f32,
  sceneFrac: f32,
  split: f32,
  errScale: f32,
};
@group(0) @binding(0) var<uniform> U: RU;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var texA: texture_2d<f32>;
@group(0) @binding(3) var texB: texture_2d<f32>;
@group(0) @binding(4) var fieldF: texture_2d<f32>;
@group(0) @binding(5) var fieldG: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> stats: array<u32, 4>;

fn A(p: vec2f) -> vec3f { return textureSampleLevel(texA, samp, p, 0.0).rgb; }
fn B(p: vec2f) -> vec3f { return textureSampleLevel(texB, samp, p, 0.0).rgb; }
fn F(p: vec2f) -> vec4f { return textureSampleLevel(fieldF, samp, p * U.fieldScale, 0.0); }
fn G(p: vec2f) -> vec4f { return textureSampleLevel(fieldG, samp, p * U.fieldScale, 0.0); }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

fn hue(h: f32) -> vec3f {
  return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

fn interp(p: vec2f, t: f32) -> vec3f {
  var v = F(p).xy;
  for (var i = 0; i < 3; i++) { v = F(p - t * v).xy; }
  var w = G(p).xy;
  for (var i = 0; i < 3; i++) { w = G(p - (1.0 - t) * w).xy; }
  let fq = F(p - t * v);
  let gq = G(p - (1.0 - t) * w);
  let a0 = A(p - t * v);
  let b0 = B(p + (1.0 - t) * v);
  let b1 = B(p - (1.0 - t) * w);
  let a1 = A(p + t * w);
  let eF = abs(lum(a0) - lum(b0)) + 0.5 * fq.z;
  let eG = abs(lum(a1) - lum(b1)) + 0.5 * gq.z;
  let k = U.errScale;
  let wF = (1.0 - t + 0.05) * exp(-eF * k);
  let wG = (t + 0.05) * exp(-eG * k);
  var col = (wF * mix(a0, b0, t) + wG * mix(a1, b1, t)) / max(wF + wG, 1e-6);
  // Occlusion: neither direction matches -> take the temporally nearer, one-sided sample.
  let bad = smoothstep(0.06, 0.16, min(eF, eG));
  var side: vec3f;
  if (t < 0.5) { side = select(a1, a0, fq.z <= gq.z); } else { side = select(b1, b0, fq.z <= gq.z); }
  col = mix(col, side, bad);
  // Optional artifact masking: fall back towards a plain blend where vectors are unreliable.
  if (U.masking > 0.0) {
    let u = smoothstep(0.03, 0.12, min(fq.z, gq.z));
    col = mix(col, mix(A(p), B(p), t), U.masking * u);
  }
  return col;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let p = (in.pos.xy - U.rect.xy) / U.rect.zw;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { return vec4f(0.0); }
  let t = U.t;
  var mode = U.mode;
  if (mode == 0u && stats[1] > 0u && f32(stats[0]) > U.sceneFrac * f32(stats[1])) { mode = 1u; }
  if (mode == 0u && in.pos.x < U.split) { mode = 1u; }
  var col: vec3f;
  switch mode {
    case 1u: { col = A(p); }
    case 2u: { col = B(p); }
    case 3u: { col = mix(A(p), B(p), t); }
    case 4u: {
      let v = F(p).xy;
      let m = clamp(length(v) * 25.0, 0.0, 1.0);
      col = mix(vec3f(lum(A(p))), hue(atan2(v.y, v.x) / 6.2831853 + 0.5), m * 0.8);
    }
    default: { col = interp(p, t); }
  }
  if (U.split > 0.0 && abs(in.pos.x - U.split) < 1.0) { col = vec3f(1.0); }
  return vec4f(col, 1.0);
}
`;

  S.WGSL = { ingest, luma, downsample, motion, finalize, render };
})();
