// Draws the state onto the page. This is "data -> DOM" only: no requests, no state kept.
//
// Two kinds of data:
//   view      full state (GET /api/state): session + ledger. Changes rarely; redrawn only when rev changes.
//   guidance  the fast loop's output for the latest frame (the reply of POST /api/frame). Redrawn every frame.

const CHECKER = { yolo: 'YOLO', pose: 'Pose', sensor: 'Sensor', cosmos: 'Cosmos', none: 'Text only' };
const SOURCE = { plan: 'Plan', yolo: 'YOLO', cosmos: 'Cosmos', sensor: 'Sensor' };
const STATUS = { todo: 'To do', passed: 'Passed', retake: 'Retake' };
const WRITER = { planner: 'Planner', slow_loop: 'Slow loop' };
const FIELD = { position: 'position', description: 'description' };
const MARK = { ok: '✓', violated: '✕', unknown: '?', degraded: '-', pending: '...', idle: '-' };

// Small helper for building DOM nodes. Text always goes through textContent (descriptions from the semantic model are external text and must never be inserted as HTML).
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'data') Object.assign(node.dataset, value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false || child === '') continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

const stamp = (source, suffix = '') =>
  h('span', { class: 'stamp', data: { source } }, (SOURCE[source] ?? source) + suffix);

function currentRecord(view) {
  return view.ledger.shots.find((record) => record.shot.id === view.current_shot_id) ?? null;
}

// Replace a set of child nodes; rows whose content differs from last time flash once (the line just written in the ledger)
function replaceRows(container, rows) {
  const before = new Map([...container.children].map((el) => [el.dataset.key, el.dataset.sig]));
  const firstPaint = container.dataset.painted !== 'yes';
  for (const row of rows) {
    if (!firstPaint && row.dataset.key && before.get(row.dataset.key) !== row.dataset.sig) {
      row.classList.add('fresh');
    }
  }
  container.replaceChildren(...rows);
  container.dataset.painted = 'yes';
}

// ───────────── Monitor side ─────────────

export function renderNow(els, view) {
  const record = currentRecord(view);
  if (!record) {
    els.shotId.textContent = '';
    els.shotTitle.textContent = view.phase === 'done' ? 'All shots passed' : 'No shot left to shoot';
    els.shotBrief.textContent = view.phase === 'done' ? 'To reshoot a shot, click "Reshoot" in the ledger on the right.' : '';
    return;
  }
  const { shot } = record;
  els.shotId.textContent = `${shot.id} · ${shot.setup}`;
  els.shotTitle.textContent = shot.title;
  els.shotBrief.textContent = `${shot.description} ${shot.intent}`;
  els.shotBrief.title = els.shotBrief.textContent;
}

export function renderRig(els, view, stats) {
  const runtime = view.runtime;
  const parts = [`Detector: ${runtime.detector}`, `Judge: ${runtime.judge}`];
  if (stats.latencyMs !== null) parts.push(`round trip ${Math.round(stats.latencyMs)} ms`);
  if (stats.fps !== null) parts.push(`${stats.fps.toFixed(1)} fps`);
  const items = parts.map((part) => h('span', {}, part));
  // Things the user should know, like a degraded detector, are not hidden in the log
  if (runtime.detector_note && runtime.source?.kind !== 'synthetic') {
    items.push(h('span', { class: 'rig-note' }, runtime.detector_note));
  }
  els.rigLine.replaceChildren(...items);
}

// Result of the last take (passed / failed / error); the current phase when there is none yet. This row always keeps its place.
export function renderNotice(els, view) {
  const notice = view.notice;
  els.notice.dataset.kind = notice ? notice.kind : 'status';
  els.notice.textContent = notice ? notice.text : view.banner;
}

export function renderCue(els, view, live) {
  const guidance = live.guidance;
  let line = 'Waiting for a picture';
  let level = 'info';
  if (view.phase === 'done') {
    [line, level] = ['All shots passed', 'ok'];
  } else if (view.phase === 'planning') {
    line = view.banner;
  } else if (view.phase === 'checking') {
    const checking = view.checking;
    line = checking
      ? `Checking ${checking.take_id} (${checking.frames} frames): ${view.runtime.judge} is judging`
      : view.banner;
  } else if (guidance) {
    [line, level] = [guidance.headline, guidance.level];
  }
  els.cue.dataset.level = level;
  els.cueLine.textContent = line;

  // Guidance derived from the ledger: a small tag in the ledger paper's colour in front, matching the ledger on the right
  els.briefing.replaceChildren(...(view.phase === 'done' ? [] : view.briefing).map((text) =>
    h('li', {}, h('span', { class: 'from-ledger' }, 'ledger'), h('span', {}, text))));
  // Requirements that cannot be checked automatically (the source / detector lacks a capability, so they became text prompts): listed one by one for the user to confirm
  const showLive = view.phase === 'framing' || view.phase === 'recording';
  const selfChecks = showLive ? (guidance?.checks ?? []).filter((check) => check.state === 'degraded' && check.message) : [];
  els.selfChecks.replaceChildren(...selfChecks.map((check) =>
    h('li', {}, h('span', { class: 'self-check' }, 'self-check'), h('span', {}, check.message))));
  els.notes.replaceChildren(...(showLive ? guidance?.notes ?? [] : []).map((text) =>
    h('li', {}, h('span', { class: 'heads-up' }, 'note'), h('span', {}, text))));
}

export function renderChecks(els, view, live) {
  const record = currentRecord(view);
  if (!record) {
    els.checks.replaceChildren();
    return;
  }
  const results = new Map((live.guidance?.checks ?? []).map((check) => [check.constraint_id, check]));
  const rows = record.shot.constraints.map((constraint) => {
    const mode = view.modes[constraint.id]?.mode ?? 'check';
    const result = results.get(constraint.id);
    const state = result?.state ?? { slow: 'pending', degraded: 'degraded' }[mode] ?? 'idle';   // idle: no picture to check right now

    const side = [];
    if (result?.readout) side.push(result.readout);
    if (result && Object.values(result.basis).includes('ledger')) side.push('from ledger');
    if (mode === 'degraded') side.push(`text only: ${view.modes[constraint.id].note}`);
    if (mode === 'slow') side.push('judged after the take');

    return h('li', { class: 'check', data: { state } },
      h('span', { class: 'check-mark', 'aria-label': state }, MARK[state] ?? '?'),
      h('span', {}, constraint.text),
      h('span', { class: 'check-side' }, side.join(', ')),
      h('span', { class: 'check-who' }, CHECKER[constraint.checker] ?? constraint.checker));
  });
  els.checks.replaceChildren(...rows);
}

export function renderControls(els, view, live, hasSource) {
  const button = els.btnTake;
  const phase = live.phase ?? view.phase;
  const set = (text, mode, enabled) => {
    button.textContent = text;
    button.dataset.mode = mode;
    button.disabled = !enabled;
  };
  if (phase === 'recording') set('End take', 'recording', true);
  else if (phase === 'checking') set('Checking', 'idle', false);
  else if (phase === 'done') set('All done', 'idle', false);
  else if (phase === 'planning') set('Writing list', 'idle', false);
  else set('Roll', live.guidance?.ready ? 'ready' : 'idle', hasSource);

  els.viewport.dataset.phase = phase;
  els.veil.hidden = phase !== 'checking';

  const take = phase === 'recording' ? live.take : null;
  els.rec.hidden = !take;
  if (take) els.recText.textContent = `REC ${take.elapsed_s.toFixed(1)} / ${take.duration_s} s`;

  els.mock.hidden = !view.runtime.judge_is_mock;
  els.mockFail.checked = view.runtime.judge_forced === false;
}

// The "Change the script" block: explains how to configure a model when there is none; no changes while recording, checking or writing
export function renderScript(els, view, busy) {
  const model = view.runtime.plan_model;
  const idle = view.phase === 'framing' || view.phase === 'done';
  els.scriptText.disabled = !model || busy;
  els.btnPlan.disabled = !model || busy || !idle;
  els.btnExample.disabled = busy || !idle;
  els.btnExample.hidden = !view.ledger.request;          // already on the example list: nothing to go back to
  els.btnPlan.textContent = busy ? 'Writing...' : 'Write the shot list';
  if (!model && !busy) {
    els.scriptStatus.dataset.kind = 'hint';
    els.scriptStatus.textContent = 'No model configured yet: set the OPENAI_API_KEY environment variable and restart run.py to write shot lists here.';
  }
}

// Draws detection boxes on the picture. Solid = detected live in this frame; dashed (ledger paper colour) = not visible, recovered from the ledger.
export function drawOverlay(canvas, guidance, names) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!guidance) return;

  const faint = (b) => {
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(241, 243, 238, 0.45)';
    ctx.strokeRect(b.x1 * width, b.y1 * height, (b.x2 - b.x1) * width, (b.y2 - b.y1) * height);
  };
  const box = (b, label, color, dashed) => {
    const x = b.x1 * width, y = b.y1 * height, w = (b.x2 - b.x1) * width, hgt = (b.y2 - b.y1) * height;
    ctx.setLineDash(dashed ? [7, 5] : []);
    ctx.lineWidth = 4;                               // a dark underlay first, so it reads on bright backgrounds too
    ctx.strokeStyle = 'rgba(31, 35, 38, 0.75)';
    ctx.strokeRect(x, y, w, hgt);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, hgt);
    ctx.setLineDash([]);
    ctx.font = '600 13px system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
    const textWidth = ctx.measureText(label).width + 10;
    const chipX = Math.max(0, Math.min(x, width - textWidth));
    const chipY = y >= 20 ? y - 20 : y + 2;
    ctx.fillStyle = color;
    ctx.fillRect(chipX, chipY, textWidth, 18);
    ctx.fillStyle = '#1f2326';
    ctx.fillText(label, chipX + 5, chipY + 13);
  };

  // With several detections of one kind, the fast loop uses the most confident: that one gets a thick labelled box, the rest thin boxes
  const used = new Map();
  for (const detection of guidance.observation.detections) {
    const best = used.get(detection.label);
    if (!best || detection.conf > best.conf) used.set(detection.label, detection);
  }
  for (const detection of guidance.observation.detections) {
    if (used.get(detection.label) !== detection) faint(detection.box);
  }
  for (const detection of used.values()) {
    box(detection.box, names[detection.label] ?? detection.label, '#f1f3ee', false);
    // Facing is judged from these points (spatial.facing_from_keypoints); drawing them helps tuning: nose yellow, ears and eyes white (eyes smaller)
    for (const point of ['left_ear', 'right_ear', 'left_eye', 'right_eye', 'nose']) {
      const [x, y, confidence] = detection.keypoints?.[point] ?? [];
      if (confidence >= 0.3) {
        ctx.fillStyle = point === 'nose' ? '#f5b82e' : '#f1f3ee';
        ctx.beginPath();
        ctx.arc(x * width, y * height, point.endsWith('eye') ? 2.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  for (const recalled of guidance.recalled ?? []) {
    box(recalled.box, `${names[recalled.label] ?? recalled.label} (ledger, ${recalled.take_id})`, '#e6eddf', true);
  }
}

// ───────────── Ledger side ─────────────

export function renderLedger(els, view, actions) {
  const { ledger } = view;
  const names = ledger.subjects;
  els.ledgerStyle.textContent = ledger.style;
  els.ledgerSynopsis.textContent = ledger.synopsis;
  els.ledgerSynopsis.hidden = !ledger.synopsis;
  els.ledgerSynopsis.title = ledger.request ? `Written from: ${ledger.request}` : '';
  els.tallyBoxes.textContent = ledger.shots.map((r) => (r.status === 'passed' ? '☑' : '☐')).join('');
  els.tallyText.textContent = `${view.coverage.passed} / ${view.coverage.total} passed`;

  const canReshoot = view.phase === 'framing' || view.phase === 'done';

  replaceRows(els.shots, ledger.shots.map((record) => {
    const { shot, effective_take: take } = record;
    const isCurrent = shot.id === view.current_shot_id;
    // A shot the user chose to reshoot: offer a way back to the shot the automatic order picks
    const reshooting = isCurrent && view.manual_selection;
    const row = h('li', {
      class: 'shot',
      data: { key: shot.id, sig: `${record.status}|${take?.take_id ?? ''}|${record.attempts}`, current: isCurrent },
    },
      h('span', { class: 'shot-id' }, shot.id),
      h('div', { class: 'shot-head' },
        h('p', { class: 'shot-title' }, shot.title, h('span', { class: 'shot-setup' }, shot.setup)),
        h('span', { class: 'shot-status', data: { status: record.status } },
          isCurrent && record.status === 'todo' ? 'Shooting' : STATUS[record.status],
          reshooting && record.status === 'passed' && ', reshooting',
          reshooting && canReshoot &&
            h('button', { type: 'button', class: 'btn-quiet cancel-reshoot', onclick: () => actions.cancelReshoot() },
              'Cancel reshoot'))));

    if (take) {
      row.append(
        h('div', { class: 'take-line' },
          h('span', { class: 'take-id', data: { kept: take.passed }, title: take.passed ? 'Kept take' : 'Failed, kept for now' },
            take.take_id),
          h('img', { class: 'take-thumb', alt: `Frame from ${take.take_id}`, loading: 'lazy',
            src: `/api/takes/${encodeURIComponent(take.take_id)}/thumb?v=${take.settled_at}` }),
          h('span', {}, `${take.n_frames} frames, ${record.attempts} ${record.attempts === 1 ? 'take' : 'takes'} shot`),
          canReshoot && record.status === 'passed' && !isCurrent &&
            h('button', { type: 'button', class: 'btn-quiet retake', onclick: () => actions.reshoot(shot.id) }, 'Reshoot')),
        h('ul', { class: 'verdicts' }, take.verdicts.map((verdict) =>
          h('li', { class: 'verdict' },
            h('span', { class: 'verdict-mark', data: { passed: String(verdict.passed) } },
              verdict.passed === true ? '✓' : verdict.passed === false ? '✕' : '–'),
            h('span', {}, verdict.text,
              verdict.detail && h('span', { class: 'verdict-detail' }, ` (${verdict.detail})`)),
            stamp(verdict.source)))));
      if (take.description) {
        row.append(h('p', { class: 'take-note' }, take.description, ' ', stamp('cosmos', `: ${take.judge}`)));
      }
    }
    return row;
  }));

  const confirm = new Set(view.confirm);
  const facts = [...ledger.scene].sort((a, b) =>
    `${a.subject}${a.field === 'position' ? 0 : 1}`.localeCompare(`${b.subject}${b.field === 'position' ? 0 : 1}`));
  replaceRows(els.facts, facts.length === 0
    ? [h('li', { class: 'empty' }, 'Nothing recorded yet. Once a shot passes, positions and descriptions of objects are written here.')]
    : facts.map((fact) => h('li', {
      class: 'fact', data: { key: `${fact.subject}.${fact.field}`, sig: `${fact.label}|${fact.take_id}` },
    },
      h('span', { class: 'fact-key' }, `${names[fact.subject] ?? fact.subject} ${FIELD[fact.field] ?? fact.field}`),
      h('span', { class: 'fact-value' }, fact.label,
        h('span', { class: 'fact-trace' },
          `from ${fact.take_id}, confidence ${fact.confidence.toFixed(2)}`,
          confirm.has(`${fact.subject}.${fact.field}`) && h('span', { class: 'fact-confirm' }, ' please confirm'))),
      stamp(fact.source))));

  replaceRows(els.log, view.log.map((entry) => h('li', {
    class: 'log-entry',
    data: { key: `${entry.version}|${entry.kind}|${entry.summary}`, sig: entry.note, applied: entry.applied },
  },
    h('span', { class: 'log-version' }, `v${entry.version}`),
    h('span', { class: 'log-writer' }, WRITER[entry.writer] ?? entry.writer),
    h('span', {}, entry.summary, h('span', { class: 'log-note' }, ` (${entry.note})`)))));
}
