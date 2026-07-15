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
  <h1>Zonos2</h1>
  <p class="sub">Pick the model files to use. They are saved for the next launch.</p>

  <label>Backbone model (.gguf)</label>
  <div class="row"><input type="text" id="model" placeholder="zonos2-q8_0.gguf">
    <button onclick="browse('model')">Browse&hellip;</button></div>

  <label>DAC decoder (.gguf)</label>
  <div class="row"><input type="text" id="dac" placeholder="dac.gguf">
    <button onclick="browse('dac')">Browse&hellip;</button></div>

  <label>Speaker encoder (.gguf, optional &mdash; enables voice cloning)</label>
  <div class="row"><input type="text" id="spk" placeholder="spk-encoder.gguf">
    <button onclick="browse('spk')">Browse&hellip;</button></div>

  <div class="check"><input type="checkbox" id="gpu" checked><label for="gpu" style="margin:0">Use GPU</label></div>

  <div class="err" id="err"></div>
  <button class="primary" id="launch" onclick="launch()">Save &amp; Launch</button>
</div>
<script>
  const $ = id => document.getElementById(id);
  function fail(msg) { $('err').textContent = msg; $('err').style.display = 'block'; }
  async function browse(kind) {
    try { const p = await saucer.exposed.pick_file(kind); if (p) $(kind).value = p; }
    catch (e) { fail(String(e)); }
  }
  async function launch() {
    $('err').style.display = 'none';
    const model = $('model').value.trim(), dac = $('dac').value.trim(), spk = $('spk').value.trim();
    if (!model || !dac) { fail('Backbone model and DAC decoder are required.'); return; }
    $('launch').disabled = true;
    try {
      const err = await saucer.exposed.launch(model, dac, spk, $('gpu').checked);
      if (err) { fail(err); $('launch').disabled = false; }
    } catch (e) { fail(String(e)); $('launch').disabled = false; }
  }
  (async () => {
    try {
      const cfg = JSON.parse(await saucer.exposed.get_config());
      if (cfg.model) $('model').value = cfg.model;
      if (cfg.dac)   $('dac').value   = cfg.dac;
      if (cfg.spk)   $('spk').value   = cfg.spk;
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
