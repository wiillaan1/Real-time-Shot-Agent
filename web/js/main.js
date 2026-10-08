// The UI's controller: pick a video source -> register -> send frames in a loop -> draw the replies; buttons just call endpoints.
//
// The frame loop waits for the previous reply before sending the next frame, so only the latest picture is ever processed:
// when the server is slow this side sends less, and nothing queues up. Every frame reply carries a rev;
// when rev changes, the session or the ledger has something new, and only then is the full state fetched to redraw the ledger.

import { api } from './api.js';
import { SimSource, WebcamSource } from './sources.js';
import {
  drawOverlay, renderChecks, renderControls, renderCue, renderLedger, renderNotice, renderNow, renderRig,
  renderScript,
} from './render.js';

const $ = (id) => document.getElementById(id);
const els = {
  monitor: $('monitor'), monitorTop: $('monitor-top'),
  shotId: $('shot-id'), shotTitle: $('shot-title'), shotBrief: $('shot-brief'),
  sourceSelect: $('source-select'), rigLine: $('rig-line'),
  viewport: $('viewport'), video: $('video'), simImage: $('sim-image'), overlay: $('overlay'),
  rec: $('rec'), recText: $('rec-text'), veil: $('veil'),
  viewportMessage: $('viewport-message'), viewportMessageText: $('viewport-message-text'), btnUseSim: $('btn-use-sim'),
  notice: $('notice'), cue: $('cue'), cueLine: $('cue-line'), briefing: $('briefing'), selfChecks: $('self-checks'),
  notes: $('notes'),
  checks: $('checks'), btnTake: $('btn-take'), mock: $('mock'), mockFail: $('mock-fail'),
  simPanel: $('sim-panel'),
  ledgerStyle: $('ledger-style'), ledgerSynopsis: $('ledger-synopsis'),
  script: $('script'), scriptText: $('script-text'), scriptStatus: $('script-status'),
  btnPlan: $('btn-plan'), btnExample: $('btn-example'),
  tallyBoxes: $('tally-boxes'), tallyText: $('tally-text'), btnReset: $('btn-reset'),
  shots: $('shots'), facts: $('facts'), log: $('log'),
};

let view = null;                                  // full state (session + ledger)
let live = { phase: null, take: null, guidance: null };   // what the latest frame reply brought
let source = null;
let upload = { fps: 3, width: 640, jpeg_quality: 0.7 };   // set by the server at registration
let loopId = 0;                                   // lets the old frame loop stop when the source changes
let lastReplyAt = 0;                              // when the last frame reply arrived
const stats = { latencyMs: null, fps: null };

// ───────────── Drawing ─────────────

function paintLive() {
  if (!view) return;
  renderCue(els, view, live);
  renderChecks(els, view, live);
  renderControls(els, view, live, source !== null);
  renderScript(els, view, planning);
  renderRig(els, view, stats);
  drawOverlay(els.overlay, view.phase === 'checking' ? null : live.guidance, view.ledger.subjects);
}

function applyView(next) {
  const changed = !view || view.rev !== next.rev;
  view = next;
  live.phase = next.phase;
  live.take = next.take;
  if (['checking', 'done', 'planning'].includes(next.phase)) live.guidance = null;
  if (changed) {                                  // the ledger side is only redrawn when something really changed
    renderNow(els, view);
    renderNotice(els, view);
    renderLedger(els, view, { reshoot, cancelReshoot });
  }
  paintLive();
}

async function refreshView() {
  try {
    applyView(await api.state());
  } catch (error) {
    offline(error);
  }
}

function offline(error) {
  els.cue.dataset.level = 'fix';
  els.cueLine.textContent = `Cannot reach the server (${error.message}). Check that run.py is still running; the page reconnects by itself`;
}

function showViewportMessage(text) {
  els.viewportMessageText.textContent = text;
  els.viewportMessage.hidden = !text;
}

// ───────────── Video source and frame loop ─────────────

function readSimScene() {
  return {
    table: $('sim-table').checked,
    bag: $('sim-bag-on').checked ? Number($('sim-bag').value) : null,
    bagUnder: $('sim-bag-under').checked,
    person: $('sim-person-on').checked ? Number($('sim-person').value) : null,
    facing: $('sim-facing').value,
    pitch: Number($('sim-pitch').value),
  };
}

async function useSource(kind) {
  loopId += 1;
  source?.close();
  source = null;
  live.guidance = null;
  showViewportMessage('');
  els.sourceSelect.value = kind;
  localStorage.setItem('shotagent.source', kind);

  const isSim = kind === 'sim';
  els.simPanel.hidden = !isSim;
  els.simImage.hidden = !isSim;
  els.video.hidden = isSim;

  const candidate = isSim ? new SimSource(els.simImage, readSimScene) : new WebcamSource(els.video);
  try {
    await candidate.open();
  } catch (error) {
    const why = error.name === 'NotAllowedError' ? 'Camera permission was not granted'
      : error.name === 'NotFoundError' ? 'No camera found' : error.message;
    showViewportMessage(`${why}. You can walk through the flow with the simulated picture first.`);
    paintLive();
    return;
  }
  source = candidate;
  els.viewport.style.setProperty('--aspect', source.aspect());
  await register().catch(offline);                // even if unreachable now, enter the frame loop: it keeps retrying and registers again once connected
  frameLoop(loopId);
}

async function register() {
  const reply = await api.registerSource(source.caps());
  upload = reply.upload;
  applyView(reply.state);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function frameLoop(id) {
  while (id === loopId && source) {
    const started = performance.now();
    try {
      const frame = await source.grab(upload);
      if (!frame) {                               // the source has no picture at this moment: send nothing, try again shortly
        await sleep(200);
        continue;
      }
      const reply = await api.pushFrame(frame);
      if (id !== loopId) return;
      onFrameReply(reply, performance.now() - started);
    } catch (error) {
      if (id !== loopId) return;
      if (error.code === 'no_source') {
        await register().catch(offline);          // the server restarted: introduce ourselves again
      } else {
        offline(error);
        await sleep(1000);
      }
    }
    const spent = performance.now() - started;
    await sleep(Math.max(0, 1000 / upload.fps - spent));
    const period = performance.now() - started;
    stats.fps = stats.fps === null ? 1000 / period : stats.fps * 0.8 + (1000 / period) * 0.2;
  }
}

function onFrameReply(reply, roundTripMs) {
  stats.latencyMs = roundTripMs;
  lastReplyAt = performance.now();
  if (reply.dropped) return;
  live = { phase: reply.phase, take: reply.take, guidance: reply.guidance };
  if (view && reply.rev !== view.rev) refreshView();   // the session or the ledger has something new
  else paintLive();
}

// ───────────── Actions ─────────────

async function act(request) {
  try {
    applyView(await request());
  } catch (error) {                               // this action is not allowed right now: the server's explanation goes in the result row
    els.notice.dataset.kind = 'error';
    els.notice.textContent = error.message;
  }
}

function toggleTake() {
  if (els.btnTake.disabled) return;
  const recording = (live.phase ?? view?.phase) === 'recording';
  act(recording ? api.stopTake : api.startTake);
}

function reshoot(shotId) {
  act(() => api.selectShot(shotId));
}

function cancelReshoot() {
  act(() => api.selectShot(null));                // null = back to the shot the automatic order picks
}

// Change the shot list. request is an idea or a script; an empty string = back to the hand-written example (no model involved).
// Generation takes ten seconds or more, so "waiting" and "failed" are shown here instead of going through act().
let planning = false;

async function replan(request) {
  if (planning) return;
  const started = view?.ledger.shots.some((record) => record.attempts > 0);
  if (started && !window.confirm('Changing the shot list clears the progress already in the ledger. Continue?')) return;

  planning = true;
  const model = view?.runtime.plan_model;
  els.scriptStatus.dataset.kind = 'busy';
  els.scriptStatus.textContent = request ? `Asking ${model} to break it into shots, usually ten seconds or more...` : '';
  paintLive();
  try {
    applyView(await api.plan(request));
    els.scriptStatus.textContent = '';
    els.script.open = false;                      // done: collapse it, the new list is right below
  } catch (error) {                               // failed: the ledger is untouched, the reason stays here, and so does the typed text
    els.scriptStatus.dataset.kind = 'error';
    els.scriptStatus.textContent = error.message;
    await refreshView();
  } finally {
    planning = false;
    paintLive();
  }
}

// ───────────── Layout ─────────────

// Shot title, picture, last result, hint and roll button must always fit on one screen: the picture gives up whatever height the others need.
// On a tall screen the constraint list is counted in as well; otherwise the picture comes first and the list sits below, reachable by scrolling.
// Only the picture's height is computed here (the CSS variable --stage-h); see .viewport in style.css for how it is used.
// These blocks keep their height within a shot (the hint line always reserves two lines), so the picture does not grow and shrink with every frame's hint.
const FIRST_SCREEN = [els.monitorTop, els.notice, els.cue];

function fitViewport() {
  if (window.matchMedia('(max-width: 900px)').matches) {     // narrow screens scroll the whole page, nothing to squeeze
    els.viewport.style.removeProperty('--stage-h');
    return;
  }
  const pane = getComputedStyle(els.monitor);
  const gap = parseFloat(pane.rowGap) || 0;
  const inner = els.monitor.clientHeight - parseFloat(pane.paddingTop) - 16;   // 16: a little air at the bottom
  const roomBeside = (blocks) => inner - blocks
    .filter((block) => block.offsetHeight > 0)
    .reduce((sum, block) => sum + block.offsetHeight + gap, 0);

  // The goal is to fit the list on this screen too, but never by pushing the picture below 36% of the screen height for the list,
  // nor below 26% for anything; at most 46%. So more text only ever makes the picture smaller, never suddenly larger.
  const screen = window.innerHeight;
  const floor = Math.max(0.26 * screen, Math.min(0.36 * screen, roomBeside(FIRST_SCREEN)));
  const height = Math.min(Math.max(roomBeside([...FIRST_SCREEN, els.checks]), floor), 0.46 * screen);
  els.viewport.style.setProperty('--stage-h', `${Math.round(height)}px`);
}

function bind() {
  els.btnTake.addEventListener('click', toggleTake);
  document.addEventListener('keydown', (event) => {
    const typing = ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY'].includes(event.target.tagName);
    if (event.code === 'Space' && !typing) {
      event.preventDefault();
      toggleTake();
    }
  });
  els.sourceSelect.addEventListener('change', () => useSource(els.sourceSelect.value));
  els.btnUseSim.addEventListener('click', () => useSource('sim'));
  els.btnReset.addEventListener('click', () => {
    if (window.confirm('Clear the progress in the ledger and start again from the first shot?')) act(api.reset);
  });
  els.btnPlan.addEventListener('click', () => {
    const request = els.scriptText.value.trim();
    if (request) replan(request);
    else els.scriptText.focus();
  });
  els.btnExample.addEventListener('click', () => replan(''));
  els.mockFail.addEventListener('change', () => act(() => api.mockNext(els.mockFail.checked ? 'fail' : null)));
  $('sim-pitch').addEventListener('input', () => { $('sim-pitch-out').textContent = `${$('sim-pitch').value}°`; });
  window.addEventListener('resize', paintLive);

  // The hint wrapping to two lines, the result row appearing, a new shot...: whenever one of these blocks changes height, the picture's height is reassigned
  const watcher = new ResizeObserver(fitViewport);
  [els.monitor, els.checks, ...FIRST_SCREEN].forEach((block) => watcher.observe(block));
  new ResizeObserver(paintLive).observe(els.viewport);       // when the picture changes size, the detection boxes are redrawn
}

async function boot() {
  bind();
  await refreshView();
  const wanted = new URLSearchParams(location.search).get('source') || localStorage.getItem('shotagent.source');
  await useSource(wanted === 'sim' ? 'sim' : 'webcam').catch(offline);
  // Safety net: if the frame loop has stopped (camera not open, server restarting), still look at the state every second
  setInterval(() => { if (performance.now() - lastReplyAt > 1500) refreshView(); }, 1000);
}

boot();
