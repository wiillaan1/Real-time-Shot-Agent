// The only place that talks to the server. The endpoints are described at the top of shotagent/server.py.

async function call(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { /* not JSON, never mind */ }
  if (!response.ok) {
    const error = new Error(body?.error || body?.detail || `Request failed (${response.status})`);
    error.status = response.status;
    error.code = body?.error;
    throw error;
  }
  return body;
}

const post = (path, data) => call(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data ?? {}),
});

export const api = {
  state: () => call('/api/state'),

  // A video source introduces itself. Returns { upload: upload parameters, state: full state }
  registerSource: (caps) => post('/api/source', caps),

  // Send one frame. Returns the framing hint for it and a brief status
  pushFrame: ({ blob, ts, sensors }) => call('/api/frame', {
    method: 'POST',
    body: blob,
    headers: {
      'Content-Type': 'image/jpeg',
      'X-Frame-Ts': String(ts),
      ...(sensors ? { 'X-Sensors': JSON.stringify(sensors) } : {}),
    },
  }),

  startTake: () => post('/api/shot/start'),
  stopTake: () => post('/api/shot/stop'),
  selectShot: (shotId) => post('/api/shot/select', { shot_id: shotId }),
  reset: () => post('/api/reset'),
  // Switch shot lists: request is an idea or a script, an empty string = back to the example. Waits on the model, ten seconds or more
  plan: (request) => post('/api/plan', { request }),
  mockNext: (next) => post('/api/debug/mock-judge', { next }),
};
