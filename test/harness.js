// Test harness: frame-accurate interpolation quality test (PSNR against the real in-between frame).
(() => {
  const S = globalThis.NNF;
  const $ = (id) => document.getElementById(id);
  const log = (...a) => {
    $('log').textContent += a.join(' ') + '\n';
  };

  function nextFrame(video, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('rVFC timeout')), timeout);
      video.requestVideoFrameCallback((now, md) => {
        clearTimeout(timer);
        resolve(md);
      });
    });
  }

  // Loaded through a blob: URL because http.server has no Range support (the clip would not be seekable).
  async function loadClip(video, src) {
    video.pause();
    if (video.src.startsWith('blob:')) URL.revokeObjectURL(video.src);
    const blob = await (await fetch(src)).blob();
    video.src = URL.createObjectURL(blob);
    await new Promise((res, rej) => {
      video.onloadeddata = res;
      video.onerror = () => rej(new Error('load failed: ' + src));
    });
  }

  // Seek to frame k (fps known) and run fn(video, md) inside the frame callback.
  async function atFrame(video, k, fps, fn) {
    const p = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('seek timeout @' + k)), 10000);
      video.requestVideoFrameCallback((now, md) => {
        clearTimeout(timer);
        try {
          resolve(fn(md));
        } catch (e) {
          reject(e);
        }
      });
    });
    video.currentTime = (k + 0.5) / fps;
    return p;
  }

  async function readTexture(device, tex, w, h) {
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.destroy();
    return out;
  }

  function psnr(a, b, border) {
    // RGB PSNR, ignoring a border (edge handling is not what we are measuring)
    const w = border.w;
    const h = border.h;
    const m = border.m;
    let se = 0;
    let n = 0;
    for (let y = m; y < h - m; y++) {
      for (let x = m; x < w - m; x++) {
        const i = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) {
          const d = a[i + c] - b[i + c];
          se += d * d;
        }
        n += 3;
      }
    }
    const mse = se / n;
    return mse === 0 ? 99 : 10 * Math.log10((255 * 255) / mse);
  }

  // For each clip: input = even frames, target = odd frames (t = 0.5).
  async function runQuality(opt = {}) {
    const clips = opt.clips || ['pan', 'fastpan', 'object', 'zoom', 'testsrc'];
    const frames = opt.frames || 40;
    const fps = 60;
    const device = await S.getDevice();
    const errors = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    const engine = new S.Interpolator(device, { ...(S.PRESETS[opt.preset || 'standard']), ...(opt.params || {}) });
    const video = $('v');
    const results = {};
    for (const clip of clips) {
      await loadClip(video, 'clips/' + clip + '.webm');
      engine.configure(video.videoWidth, video.videoHeight);
      const r = engine.res;
      const out = device.createTexture({
        size: [r.fw, r.fh],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const outView = out.createView();
      const sums = { mc: 0, blend: 0, repeat: 0 };
      const per = [];
      let prev = null;
      let pairs = 0;
      let sceneCuts = 0;
      for (let k = 0; k + 2 <= frames; k += 2) {
        const a = prev || (await atFrame(video, k, fps, () => engine.ingest(video, null)));
        const gt = await atFrame(video, k + 1, fps, () => engine.ingest(video, null));
        const b = await atFrame(video, k + 2, fps, () => engine.ingest(video, a));
        const gtPix = await readTexture(device, gt.tex, r.fw, r.fh);
        const score = {};
        for (const [name, mode] of [['mc', 0], ['blend', 3], ['repeat', 1]]) {
          engine.render(outView, 'rgba8unorm', [0, 0, r.fw, r.fh], a, b, 0.5, { mode });
          const pix = await readTexture(device, out, r.fw, r.fh);
          score[name] = psnr(pix, gtPix, { w: r.fw, h: r.fh, m: 16 });
          sums[name] += score[name];
        }
        const st = await engine.readStats(b);
        if (st.bad > engine.params.sceneFrac * st.total) sceneCuts++;
        per.push(+score.mc.toFixed(2));
        pairs++;
        prev = b;
      }
      out.destroy();
      results[clip] = {
        mc: +(sums.mc / pairs).toFixed(2),
        blend: +(sums.blend / pairs).toFixed(2),
        repeat: +(sums.repeat / pairs).toFixed(2),
        gain: +((sums.mc - sums.blend) / pairs).toFixed(2),
        min: Math.min(...per),
        sceneCuts,
      };
      log(clip, JSON.stringify(results[clip]));
    }
    engine.destroy();
    results.errors = [...new Set(errors)].slice(0, 3).map((e) => e.slice(0, 400));
    return results;
  }

  // Render one interpolated frame of a clip to PNG data URL (for visual inspection).
  async function snapshot(opt = {}) {
    const clip = opt.clip || 'object';
    const k = opt.frame || 20;
    const t = opt.t ?? 0.5;
    const device = await S.getDevice();
    const engine = new S.Interpolator(device, { ...(S.PRESETS[opt.preset || 'standard']), ...(opt.params || {}) });
    const video = $('v');
    await loadClip(video, 'clips/' + clip + '.webm');
    engine.configure(video.videoWidth, video.videoHeight);
    const r = engine.res;
    const p0 = await atFrame(video, k - 2, 60, () => engine.ingest(video, null));
    const a = await atFrame(video, k, 60, () => engine.ingest(video, p0));
    const b = await atFrame(video, k + 2, 60, () => engine.ingest(video, a));
    const out = device.createTexture({
      size: [r.fw, r.fh],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    engine.render(out.createView(), 'rgba8unorm', [0, 0, r.fw, r.fh], a, b, t, { mode: opt.mode || 0 });
    const pix = await readTexture(device, out, r.fw, r.fh);
    const c = document.createElement('canvas');
    c.width = r.fw;
    c.height = r.fh;
    c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pix.buffer), r.fw, r.fh), 0, 0);
    out.destroy();
    engine.destroy();
    return c.toDataURL('image/png');
  }

  // Live playback through the same Player the extension uses (selected clip or a local file).
  let player = null;
  async function play() {
    const video = $('v');
    const file = $('file').files[0];
    if (file) {
      if (video.src.startsWith('blob:')) URL.revokeObjectURL(video.src);
      video.src = URL.createObjectURL(file);
    } else {
      await loadClip(video, 'clips/' + $('clip').value + '.webm');
    }
    if (player) player.destroy();
    const settings = S.mergeSettings({ demo: $('demo').checked, osd: true, audioSync: false });
    player = new S.Player(video, settings, { id: 'harness', name: 'Harness' });
    globalThis.player = player;
    await video.play();
  }
  $('demo').onchange = () => player && player.applySettings({ ...player.settings, demo: $('demo').checked });

  globalThis.runQuality = runQuality;
  globalThis.snapshot = snapshot;
  $('quality').onclick = () => runQuality().then((r) => log(JSON.stringify(r)), (e) => log('ERROR', e.message));
  $('play').onclick = () => play().catch((e) => log('ERROR', e.message));
})();
