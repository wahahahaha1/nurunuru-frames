// GPU frame interpolator: frame ingest, hierarchical block-matching motion estimation and
// motion-compensated rendering. Frames live in a ring of RING slots; the motion fields of the
// pair (previous frame -> frame) are stored with the later frame.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const RING = 6;
  const BU = GPUBufferUsage;
  const TU = GPUTextureUsage;

  class Interpolator {
    constructor(device, params) {
      this.device = device;
      this.params = { ...S.TUNING, ...S.PRESETS.standard, ...params };
      const W = S.WGSL;
      const mod = (code, label) => device.createShaderModule({ code, label });
      this.samp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      const ingestMod = mod(W.ingest, 'ingest');
      this.ingestPipe = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: ingestMod, entryPoint: 'vs' },
        fragment: { module: ingestMod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      });
      const comp = (code, label) =>
        device.createComputePipeline({ layout: 'auto', compute: { module: mod(code, label), entryPoint: 'main' } });
      this.lumaPipe = comp(W.luma, 'luma');
      this.downPipe = comp(W.downsample, 'downsample');
      this.mePipe = comp(W.motion, 'motion');
      this.finPipe = comp(W.finalize, 'finalize');
      this.renderMod = mod(W.render, 'render');
      this.renderPipes = new Map();
      this.renderU = device.createBuffer({ size: 48, usage: BU.UNIFORM | BU.COPY_DST });
      this.zeroBuf = device.createBuffer({ size: 64, usage: BU.STORAGE });
      this.zeroField = device.createTexture({ size: [1, 1], format: 'rgba16float', usage: TU.TEXTURE_BINDING });
      this.zeroStats = device.createBuffer({ size: 16, usage: BU.STORAGE });
      this.res = null;
      this.seq = 0;
      // GPU timing (sampled): onTiming('me' | 'render', ms) reports ingest+motion and render cost.
      this.onTiming = null;
      this.timing = device.features.has('timestamp-query');
      this.ingestCount = 0;
      this.renderCount = 0;
      if (this.timing) {
        this.querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
        this.resolveBuf = device.createBuffer({ size: 512, usage: BU.QUERY_RESOLVE | BU.COPY_SRC });
        this.readbacks = [0, 1, 2, 3].map(() => device.createBuffer({ size: 16, usage: BU.COPY_DST | BU.MAP_READ }));
      }
    }

    // Returns a free readback buffer if this call should be measured, else null.
    sample(counter, every) {
      if (!this.timing || !this.onTiming || counter % every !== 0) return null;
      return this.readbacks.pop() || null;
    }

    // Resolve query pair (first, first+1) into rb and report the elapsed GPU time once mapped.
    finishSample(enc, rb, first, kind) {
      const off = first === 0 ? 0 : 256;
      enc.resolveQuerySet(this.querySet, first, 2, this.resolveBuf, off);
      enc.copyBufferToBuffer(this.resolveBuf, off, rb, 0, 16);
      return () =>
        rb.mapAsync(GPUMapMode.READ).then(
          () => {
            const t = new BigInt64Array(rb.getMappedRange());
            const ms = Number(t[1] - t[0]) / 1e6;
            rb.unmap();
            this.readbacks.push(rb);
            if (ms >= 0 && ms < 1000 && this.onTiming) this.onTiming(kind, ms * (S.debugGpuScale || 1));
          },
          () => {} // device lost / destroyed
        );
    }

    setParams(params) {
      const old = this.params;
      this.params = { ...this.params, ...params };
      const geo = ['maxFrameWidth', 'meWidth', 'bs', 'ov', 'topRadius', 'radius', 'subpel', 'lambda', 'lambdaZero', 'useMedian', 'sceneSad'];
      if (this.res && geo.some((k) => old[k] !== this.params[k])) {
        const { srcW, srcH } = this.res;
        this.destroyResources();
        this.configure(srcW, srcH);
      }
    }

    // (Re)allocate everything for a source of srcW x srcH. Returns true when resources changed.
    configure(srcW, srcH) {
      if (this.res && this.res.srcW === srcW && this.res.srcH === srcH) return false;
      this.destroyResources();
      const d = this.device;
      const p = this.params;
      const fs = Math.min(1, p.maxFrameWidth / srcW);
      const fw = Math.max(16, Math.round(srcW * fs));
      const fh = Math.max(16, Math.round(srcH * fs));
      const ms = Math.min(1, p.meWidth / srcW);
      const mw = Math.max(32, Math.round(srcW * ms));
      const mh = Math.max(32, Math.round(srcH * ms));
      const levels = Math.max(2, Math.min(5, Math.floor(Math.log2(mw / 100)) + 1));
      const lw = [mw];
      const lh = [mh];
      for (let k = 1; k < levels; k++) {
        lw.push(Math.ceil(lw[k - 1] / 2));
        lh.push(Math.ceil(lh[k - 1] / 2));
      }
      const nbx = lw.map((w) => Math.ceil(w / p.bs));
      const nby = lh.map((h) => Math.ceil(h / p.bs));
      const uni = (data) => {
        const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: BU.UNIFORM | BU.COPY_DST });
        d.queue.writeBuffer(b, 0, data);
        return b;
      };
      const ingestScale = fw / srcW;
      const ingestU = uni(new Float32Array([1 / fw, 1 / fh, ingestScale < 0.75 ? 2 : 1, 0]));
      const lumaU = uni(new Uint32Array([mw, mh, Math.max(1, Math.min(4, Math.round(fw / mw))), 0]));
      const downU = [null];
      for (let k = 1; k < levels; k++) downU.push(uni(new Uint32Array([lw[k - 1], lh[k - 1], lw[k], lh[k]])));
      const meU = [];
      for (let k = 0; k < levels; k++) {
        const top = k === levels - 1;
        meU.push(
          [0, 1].map((useT) => {
            const buf = new ArrayBuffer(64);
            const u = new Uint32Array(buf);
            const i = new Int32Array(buf);
            const f = new Float32Array(buf);
            u.set([lw[k], lh[k], nbx[k], nby[k], top ? 1 : nbx[k + 1], top ? 1 : nby[k + 1], p.bs, p.ov]);
            i[8] = top ? p.topRadius : p.radius;
            u[9] = top ? 0 : 1;
            u[10] = useT;
            u[11] = k === 0 ? p.subpel : 0;
            f[12] = p.lambda;
            f[13] = p.lambdaZero;
            return uni(buf);
          })
        );
      }
      const finBuf = new ArrayBuffer(32);
      new Uint32Array(finBuf).set([nbx[0], nby[0], mw, mh, 0, p.useMedian ? 1 : 0, 0, 0]);
      new Float32Array(finBuf)[4] = p.sceneSad;
      const finU = uni(finBuf);

      const frames = [];
      for (let s = 0; s < RING; s++) {
        const tex = d.createTexture({
          size: [fw, fh],
          format: 'rgba8unorm',
          usage: TU.TEXTURE_BINDING | TU.RENDER_ATTACHMENT | TU.COPY_SRC,
        });
        const luma = lw.map((w, k) => d.createBuffer({ size: w * lh[k] * 4, usage: BU.STORAGE }));
        const vec = (k) => d.createBuffer({ size: nbx[k] * nby[k] * 16, usage: BU.STORAGE });
        const field = () =>
          d.createTexture({ size: [nbx[0], nby[0]], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING });
        const fr = {
          slot: s,
          tex,
          view: tex.createView(),
          luma,
          vecF: lw.map((_, k) => vec(k)),
          vecG: lw.map((_, k) => vec(k)),
          fieldF: field(),
          fieldG: field(),
          stats: d.createBuffer({ size: 16, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC }),
          id: -1, // sequence number of the frame currently held
          hasPair: false, // fields for (id-1 -> id) are valid
          pairPrevId: -1,
        };
        fr.fieldFView = fr.fieldF.createView();
        fr.fieldGView = fr.fieldG.createView();
        fr.lumaBG = d.createBindGroup({
          layout: this.lumaPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.samp },
            { binding: 1, resource: fr.view },
            { binding: 2, resource: { buffer: luma[0] } },
            { binding: 3, resource: { buffer: lumaU } },
          ],
        });
        fr.downBG = [null];
        for (let k = 1; k < levels; k++) {
          fr.downBG.push(
            d.createBindGroup({
              layout: this.downPipe.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: luma[k - 1] } },
                { binding: 1, resource: { buffer: luma[k] } },
                { binding: 2, resource: { buffer: downU[k] } },
              ],
            })
          );
        }
        frames.push(fr);
      }
      this.res = {
        srcW, srcH, fw, fh, mw, mh, levels, lw, lh, nbx, nby,
        ingestU, lumaU, downU, meU, finU, frames, next: 0,
        uniforms: [ingestU, lumaU, ...downU.filter(Boolean), ...meU.flat(), finU],
      };
      return true;
    }

    destroyResources() {
      if (!this.res) return;
      for (const f of this.res.frames) {
        f.tex.destroy();
        f.fieldF.destroy();
        f.fieldG.destroy();
        f.stats.destroy();
        for (const b of [...f.luma, ...f.vecF, ...f.vecG]) b.destroy();
      }
      for (const b of this.res.uniforms) b.destroy();
      this.res = null;
    }

    destroy() {
      this.destroyResources();
      this.renderU.destroy();
      this.zeroBuf.destroy();
      this.zeroField.destroy();
      this.zeroStats.destroy();
      this.onTiming = null;
      if (this.timing) {
        this.querySet.destroy();
        this.resolveBuf.destroy();
        for (const b of this.readbacks) b.destroy();
      }
    }

    // Upload a video frame (HTMLVideoElement or VideoFrame) into the next ring slot and build its
    // luma pyramid. If `prev` (a frame returned earlier) is given, motion between prev and this
    // frame is estimated in the same submission. Returns the frame record.
    ingest(source, prev, opts = {}) {
      const d = this.device;
      const r = this.res;
      const f = r.frames[r.next];
      r.next = (r.next + 1) % RING;
      f.id = ++this.seq;
      f.hasPair = false;
      f.pairPrevId = -1;
      const ext = d.importExternalTexture({ source });
      const enc = d.createCommandEncoder();
      const withMotion = prev && prev.id > 0 && this.isLive(prev);
      // measure the steady-state cost (ingest + motion) on every 4th frame
      const rb = withMotion ? this.sample(this.ingestCount++, 4) : null;
      const bg = d.createBindGroup({
        layout: this.ingestPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.samp },
          { binding: 1, resource: ext },
          { binding: 2, resource: { buffer: r.ingestU } },
        ],
      });
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: f.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
        timestampWrites: rb ? { querySet: this.querySet, beginningOfPassWriteIndex: 0 } : undefined,
      });
      rp.setPipeline(this.ingestPipe);
      rp.setBindGroup(0, bg);
      rp.draw(3);
      rp.end();
      const cp = enc.beginComputePass();
      cp.setPipeline(this.lumaPipe);
      cp.setBindGroup(0, f.lumaBG);
      cp.dispatchWorkgroups(Math.ceil(r.mw / 16), Math.ceil(r.mh / 16));
      cp.setPipeline(this.downPipe);
      for (let k = 1; k < r.levels; k++) {
        cp.setBindGroup(0, f.downBG[k]);
        cp.dispatchWorkgroups(Math.ceil(r.lw[k] / 16), Math.ceil(r.lh[k] / 16));
      }
      cp.end();
      let report = null;
      if (withMotion) {
        const tw = rb ? { querySet: this.querySet, endOfPassWriteIndex: 1 } : undefined;
        this.encodeMotion(enc, prev, f, opts.temporal !== false, tw);
        if (rb) report = this.finishSample(enc, rb, 0, 'me');
      }
      d.queue.submit([enc.finish()]);
      if (report) report();
      return f;
    }

    // A frame record stays usable until its slot is reused.
    isLive(frame) {
      return !!this.res && this.res.frames[frame.slot] === frame && frame.id > 0;
    }

    encodeMotion(enc, a, b, allowTemporal, timestampWrites) {
      const d = this.device;
      const r = this.res;
      // temporal predictor = fields of the pair ending at frame a (a-1 -> a), if still valid
      const useT = allowTemporal && a.hasPair ? 1 : 0;
      enc.clearBuffer(b.stats);
      const cp = enc.beginComputePass({ timestampWrites });
      cp.setPipeline(this.mePipe);
      const dirs = [
        { cur: a, ref: b, out: b.vecF, tmp: a.vecF },
        { cur: b, ref: a, out: b.vecG, tmp: a.vecG },
      ];
      for (const dir of dirs) {
        for (let k = r.levels - 1; k >= 0; k--) {
          const top = k === r.levels - 1;
          cp.setBindGroup(
            0,
            d.createBindGroup({
              layout: this.mePipe.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: r.meU[k][useT] } },
                { binding: 1, resource: { buffer: dir.cur.luma[k] } },
                { binding: 2, resource: { buffer: dir.ref.luma[k] } },
                { binding: 3, resource: { buffer: top ? this.zeroBuf : dir.out[k + 1] } },
                { binding: 4, resource: { buffer: useT ? dir.tmp[k] : this.zeroBuf } },
                { binding: 5, resource: { buffer: dir.out[k] } },
              ],
            })
          );
          cp.dispatchWorkgroups(r.nbx[k], r.nby[k]);
        }
      }
      cp.setPipeline(this.finPipe);
      for (const [vec, field] of [[b.vecF[0], b.fieldFView], [b.vecG[0], b.fieldGView]]) {
        cp.setBindGroup(
          0,
          d.createBindGroup({
            layout: this.finPipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: vec } },
              { binding: 1, resource: field },
              { binding: 2, resource: { buffer: b.stats } },
              { binding: 3, resource: { buffer: r.finU } },
            ],
          })
        );
        cp.dispatchWorkgroups(Math.ceil(r.nbx[0] / 8), Math.ceil(r.nby[0] / 8));
      }
      cp.end();
      b.hasPair = true;
      b.pairPrevId = a.id;
    }

    renderPipe(format) {
      let pipe = this.renderPipes.get(format);
      if (!pipe) {
        pipe = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: { module: this.renderMod, entryPoint: 'vs' },
          fragment: { module: this.renderMod, entryPoint: 'fs', targets: [{ format }] },
        });
        this.renderPipes.set(format, pipe);
      }
      return pipe;
    }

    // Draw frame a..b at phase t into `view` (format `format`). rect = content rectangle
    // [x, y, w, h] in target pixels. opts: { mode, split, masking }.
    // mode: 0 interpolate, 1 frame A, 2 frame B, 3 plain blend, 4 vectors.
    render(view, format, rect, a, b, t, opts = {}) {
      const d = this.device;
      const r = this.res;
      const p = this.params;
      if (!b || !this.isLive(b)) b = a;
      const paired = b !== a && b.hasPair && b.pairPrevId === a.id;
      let mode = opts.mode || 0;
      if (!paired && (mode === 0 || mode === 4)) mode = b === a ? 1 : 3;
      const buf = new ArrayBuffer(48);
      const f = new Float32Array(buf);
      const u = new Uint32Array(buf);
      f.set(rect, 0);
      f[4] = r.mw / (p.bs * r.nbx[0]);
      f[5] = r.mh / (p.bs * r.nby[0]);
      f[6] = t;
      u[7] = mode;
      f[8] = opts.masking ?? p.masking;
      f[9] = p.sceneFrac;
      f[10] = opts.split || 0;
      f[11] = p.errScale;
      d.queue.writeBuffer(this.renderU, 0, buf);
      const pipe = this.renderPipe(format);
      const bg = d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderU } },
          { binding: 1, resource: this.samp },
          { binding: 2, resource: a.view },
          { binding: 3, resource: b.view },
          { binding: 4, resource: paired ? b.fieldFView : this.zeroField.createView() },
          { binding: 5, resource: paired ? b.fieldGView : this.zeroField.createView() },
          { binding: 6, resource: { buffer: paired ? b.stats : this.zeroStats } },
        ],
      });
      const enc = d.createCommandEncoder();
      const rb = this.sample(this.renderCount++, 20);
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        timestampWrites: rb ? { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } : undefined,
      });
      rp.setPipeline(pipe);
      rp.setBindGroup(0, bg);
      rp.draw(3);
      rp.end();
      const report = rb ? this.finishSample(enc, rb, 2, 'render') : null;
      d.queue.submit([enc.finish()]);
      if (report) report();
    }

    // Scene-change statistics of the pair ending at frame b: { bad, total } (async, for OSD/tests).
    async readStats(b) {
      const d = this.device;
      const rb = d.createBuffer({ size: 16, usage: BU.COPY_DST | BU.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(b.stats, 0, rb, 0, 16);
      d.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const v = new Uint32Array(rb.getMappedRange().slice(0));
      rb.destroy();
      return { bad: v[0], total: v[1] };
    }
  }

  S.Interpolator = Interpolator;
})();
