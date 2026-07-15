// Built-in pages for zonos2-app, shown via saucer's webview->set_html() before the
// embedded server is up (setup / loading / error). Self-contained HTML: no network,
// no external assets. The pages call C++ through the functions app.cpp exposes on
// the smartview (`saucer.exposed.*`). ZONOS2_APP_ERROR_PAGE contains a %DETAILS%
// placeholder replaced (HTML-escaped) by app.cpp.
#pragma once

static const char * ZONOS2_APP_STYLE = R"html(
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: #101216; color: #e6e8ee;
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .card { width: min(560px, 92vw); background: #171a21; border: 1px solid #262b36;
          border-radius: 12px; padding: 28px 32px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8b93a7; font-size: 13px; margin: 0 0 22px; }
  label { display: block; font-size: 13px; color: #aab1c2; margin: 14px 0 4px; }
  .row { display: flex; gap: 8px; }
  input[type=text] { flex: 1; background: #0d0f13; color: #e6e8ee; border: 1px solid #2a3040;
                     border-radius: 7px; padding: 8px 10px; font-size: 13px; min-width: 0; }
  input[type=text]:focus { outline: none; border-color: #5b7fd4; }
  select { width: 100%; background: #0d0f13; color: #e6e8ee; border: 1px solid #2a3040;
           border-radius: 7px; padding: 8px 10px; font-size: 13px; }
  select:focus { outline: none; border-color: #5b7fd4; }
  a { color: #7fa0ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .bar { height: 8px; background: #0d0f13; border: 1px solid #2a3040; border-radius: 6px;
         overflow: hidden; margin: 4px 0 14px; }
  .barfill { height: 100%; width: 0; background: #3b5bd6; transition: width .3s ease; }
  button { background: #222836; color: #e6e8ee; border: 1px solid #323a4d; border-radius: 7px;
           padding: 8px 14px; font-size: 13px; cursor: pointer; white-space: nowrap; }
  button:hover { background: #2a3242; }
  button.primary { background: #3b5bd6; border-color: #3b5bd6; font-weight: 600;
                   width: 100%; margin-top: 22px; padding: 10px; font-size: 14px; }
  button.primary:hover { background: #4a69e2; }
  button:disabled { opacity: .5; cursor: default; }
  .check { display: flex; align-items: center; gap: 8px; margin-top: 16px; font-size: 14px; }
  .err { color: #ff8d8d; font-size: 13px; margin-top: 14px; white-space: pre-wrap; display: none; }
  .spinner { width: 44px; height: 44px; margin: 0 auto 18px; border-radius: 50%;
             border: 3px solid #2a3040; border-top-color: #5b7fd4; animation: spin 0.9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .center { text-align: center; }
  pre.details { background: #0d0f13; border: 1px solid #2a3040; border-radius: 7px; padding: 12px;
                font-size: 12px; white-space: pre-wrap; word-break: break-word; color: #c3c9d8;
                max-height: 200px; overflow: auto; text-align: left; }
</style>
)html";

// ---------------------------------------------------------------- setup (first run)

static const char * ZONOS2_APP_SETUP_PAGE = R"html(
<div class="card">
  <div id="main">
    <h1>Zonos2</h1>
    <p class="sub">Download the model files to get started, or point to files you already have.</p>

    <div id="dl-view">
      <label>Model quality</label>
      <select id="quant">
        <option value="q4_k">q4_k &mdash; smallest (4.9 GB)</option>
        <option value="q5_k">q5_k (5.8 GB)</option>
        <option value="q6_k" selected>q6_k &mdash; recommended (6.8 GB)</option>
        <option value="q8_0">q8_0 &mdash; best quality (8.5 GB)</option>
        <option value="f16">f16 &mdash; full precision (15.3 GB)</option>
      </select>
      <div class="check"><input type="checkbox" id="want_spk" checked>
        <label for="want_spk" style="margin:0">Include voice cloning (spk-encoder, +20&nbsp;MB)</label></div>
    </div>

    <div id="manual-view" style="display:none">
      <label>Backbone model (.gguf)</label>
      <div class="row"><input type="text" id="model_in" placeholder="zonos2-q6_k.gguf">
        <button onclick="browse('model')">Browse&hellip;</button></div>
      <label>DAC decoder (.gguf)</label>
      <div class="row"><input type="text" id="dac_in" placeholder="dac.gguf">
        <button onclick="browse('dac')">Browse&hellip;</button></div>
      <label>Speaker encoder (.gguf, optional &mdash; enables voice cloning)</label>
      <div class="row"><input type="text" id="spk_in" placeholder="spk-encoder.gguf">
        <button onclick="browse('spk')">Browse&hellip;</button></div>
    </div>

    <div class="check"><input type="checkbox" id="gpu" checked>
      <label for="gpu" style="margin:0">Use GPU</label></div>

    <div class="err" id="err"></div>
    <button class="primary" id="dlbtn" onclick="download()">Download &amp; Launch</button>
    <button class="primary" id="launchbtn" onclick="launchManual()" style="display:none">Save &amp; Launch</button>
    <p class="sub" style="margin:14px 0 0"><a href="#" id="toggle" onclick="toggleManual(event)">I already have the model files</a></p>
  </div>

  <div id="progress" style="display:none" class="center">
    <div class="spinner"></div>
    <h1 id="pgtitle">Downloading&hellip;</h1>
    <div class="bar"><div class="barfill" id="barfill"></div></div>
    <p class="sub" id="pgtext" style="margin:0">Fetching model files. This can take a while on first launch.</p>
  </div>
</div>
<script>
  const $ = id => document.getElementById(id);
  function fail(msg) { $('err').textContent = msg; $('err').style.display = 'block'; }
  function clearErr() { $('err').style.display = 'none'; }
  function fmtGB(b) { return (b / 1e9).toFixed(2) + ' GB'; }

  let manualOn = false;
  function toggleManual(e) {
    e.preventDefault(); manualOn = !manualOn; clearErr();
    $('dl-view').style.display     = manualOn ? 'none'  : 'block';
    $('manual-view').style.display = manualOn ? 'block' : 'none';
    $('dlbtn').style.display       = manualOn ? 'none'  : 'block';
    $('launchbtn').style.display   = manualOn ? 'block' : 'none';
    $('toggle').textContent = manualOn ? 'Download the models instead' : 'I already have the model files';
  }

  async function browse(kind) {
    try { const p = await saucer.exposed.pick_file(kind); if (p) $(kind + '_in').value = p; }
    catch (e) { fail(String(e)); }
  }

  async function doLaunch(model, dac, spk, gpu) {
    const err = await saucer.exposed.launch(model, dac, spk, gpu);
    if (err) throw new Error(err);   // C++ swaps to the splash page on success
  }

  async function launchManual() {
    clearErr();
    const model = $('model_in').value.trim(), dac = $('dac_in').value.trim(), spk = $('spk_in').value.trim();
    if (!model || !dac) { fail('Backbone model and DAC decoder are required.'); return; }
    $('launchbtn').disabled = true;
    try { await doLaunch(model, dac, spk, $('gpu').checked); }
    catch (e) { fail(String(e.message || e)); $('launchbtn').disabled = false; }
  }

  async function download() {
    clearErr();
    const quant = $('quant').value, wantSpk = $('want_spk').checked, gpu = $('gpu').checked;
    $('main').style.display = 'none'; $('progress').style.display = 'block';
    try { await saucer.exposed.download_models(quant, wantSpk, gpu); }
    catch (e) { backToSetup(String(e.message || e)); return; }
    poll();
  }

  function backToSetup(msg) {
    $('progress').style.display = 'none'; $('main').style.display = 'block';
    if (msg) fail(msg);
  }

  async function poll() {
    let s;
    try { s = JSON.parse(await saucer.exposed.download_status()); }
    catch (e) { setTimeout(poll, 600); return; }
    if (s.file) {
      const pct = s.total > 0 ? Math.min(100, 100 * s.done / s.total) : 0;
      $('barfill').style.width = pct + '%';
      $('pgtitle').textContent = 'Downloading ' + s.file;
      $('pgtext').textContent  = 'File ' + s.idx + ' of ' + s.count + ' — ' +
        fmtGB(s.done) + (s.total > 0 ? ' / ' + fmtGB(s.total) : '');
    }
    if (s.finished) {
      if (!s.ok) { backToSetup(s.error || 'download failed'); return; }
      $('pgtitle').textContent = 'Starting…'; $('barfill').style.width = '100%';
      try { await doLaunch(s.model, s.dac, s.spk, s.gpu); }
      catch (e) { backToSetup(String(e.message || e)); }
      return;
    }
    setTimeout(poll, 600);
  }

  (async () => {
    try {
      const cfg = JSON.parse(await saucer.exposed.get_config());
      if (cfg.model) $('model_in').value = cfg.model;
      if (cfg.dac)   $('dac_in').value   = cfg.dac;
      if (cfg.spk)   $('spk_in').value   = cfg.spk;
      $('gpu').checked = cfg.gpu !== false;
    } catch (e) { /* first run: no config yet */ }
  })();
</script>
)html";

// ---------------------------------------------------------------- loading splash

static const char * ZONOS2_APP_SPLASH_PAGE = R"html(
<div class="card center">
  <div class="spinner"></div>
  <h1>Loading models&hellip;</h1>
  <p class="sub" id="status" style="margin-bottom:0">This can take a little while on first launch.</p>
</div>
)html";

// ---------------------------------------------------------------- startup error

static const char * ZONOS2_APP_ERROR_PAGE = R"html(
<div class="card center">
  <h1>Failed to start</h1>
  <p class="sub">The embedded zonos2 server did not come up.</p>
  <pre class="details">%DETAILS%</pre>
  <button class="primary" onclick="saucer.exposed.show_setup()">Back to setup</button>
</div>
)html";
