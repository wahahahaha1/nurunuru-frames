// Shared WebGPU device + engine presets.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  // Motion-estimation presets. meWidth = luma width used for block matching.
  // Measured on RTX 3080, 1080p source: fast ~1.0 ms, standard ~2.2 ms, quality ~5 ms per source frame.
  S.PRESETS = {
    fast: { meWidth: 640, bs: 8, ov: 2, topRadius: 6, radius: 1, subpel: 1 },
    standard: { meWidth: 960, bs: 8, ov: 4, topRadius: 8, radius: 2, subpel: 1 },
    quality: { meWidth: 1280, bs: 8, ov: 4, topRadius: 10, radius: 2, subpel: 2 },
  };

  // Tuning shared by all presets (luma is 0..1, distances are in ME-level pixels).
  S.TUNING = {
    maxFrameWidth: 2560, // frames wider than this are downscaled on ingest
    lambda: 0.003, // penalty per pixel of deviation from the predictor
    lambdaZero: 0.0005, // slight preference for short vectors
    useMedian: true,
    sceneSad: 0.1, // block SAD/pixel above which a block counts as "unmatched"
    sceneFrac: 0.45, // fraction of unmatched blocks that marks a scene change
    errScale: 30, // sharpness of the per-pixel consistency weighting
    masking: 0, // 0..1, artifact masking towards a plain blend
  };

  S.getDevice = function getDevice() {
    if (S._devicePromise) return S._devicePromise;
    S._devicePromise = (async () => {
      if (!navigator.gpu) throw new Error('WebGPU が利用できません (navigator.gpu なし)');
      // powerPreference is ignored on Windows and only produces a console warning there (crbug.com/369219127)
      const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
      const opts = /^Win/i.test(platform) ? undefined : { powerPreference: 'high-performance' };
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (!adapter) throw new Error('WebGPU アダプターを取得できません');
      // timestamp queries let the player measure its own GPU cost (performance guard)
      const requiredFeatures = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
      const device = await adapter.requestDevice({ requiredFeatures });
      device.lost.then((info) => {
        S._devicePromise = null;
        if (S.onDeviceLost) S.onDeviceLost(info);
      });
      return device;
    })();
    S._devicePromise.catch(() => {
      S._devicePromise = null;
    });
    return S._devicePromise;
  };
})();
