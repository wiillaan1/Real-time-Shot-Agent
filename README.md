[README.md](https://github.com/user-attachments/files/33182890/README.md)
# Live Shooting Agent: skeleton

A first version built to section 3 of the design doc: the goal is a complete agent loop that runs, not a full feature set.

What it does now: the browser sends 3 frames a second → the fast loop gives framing hints → a take ends by button or timer →
the slow loop settles it → the ledger is updated → on a pass the next shot comes up, on a fail you reshoot in place.
It ships with three hand-written shots around one example, "the bag under the table".
With a model key configured you can type an idea or paste a script in the UI and have a model break it into a shot list
(see "Change the script").
The semantic judge is a mock by default; the real Cosmos client is written to the organisers' docs but has never been run
against the real endpoint.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # ultralytics is large and optional (see below)
python run.py                       # open http://localhost:8000
```

- **No camera and no YOLO needed**: switch "Video source" at the top right to "Simulated" and place the table, bag and person
  with the sliders. The whole loop really runs (YOLO is just replaced by finding blobs of colour).
- With a camera but without ultralytics: detection-based constraints fall back to text prompts, the top right of the UI says why,
  and the list can still be completed.
- A narrated run in the terminal: `python scripts/simulate.py`
- Tests: `pytest` (194; 4 of them need ultralytics and are skipped without it)

Common flags: `--detector yolo|none`, `--judge cosmos`, `--take-seconds 8` (longer takes when one person both shoots and acts),
`--yolo-device mps` (Apple silicon), `--fresh` (clear progress). Every tunable threshold is in `shotagent/config.py`.

## Change the script: idea → shot list

To film something other than the Hitchcock example, have a model write the list from your idea or script:

```bash
export OPENAI_API_KEY=sk-...        # defaults to OpenAI, model gpt-5.6-luna
python scripts/try_plan.py "Someone working late goes to get water and comes back to find a book on the desk that is not theirs"   # try it once by itself
python run.py                       # then click "Change the script" above the ledger
```

- The input can be a one-line idea, a style, or a finished script (a script is broken down as written, with no added plot).
- The model is given a **menu**: which roles the system can see (person, bag, table, chair, cup... see `config.DEFAULT_LABEL_MAP`)
  and which relations it can check automatically (the ones registered in `constraints.PREDICATES`). Anything outside the menu
  has to go under "judged by the semantic model after the take" or "text prompt only". The UI shows who checks each constraint,
  exactly as for the hand-written list.
- The model's output is never used directly: predicate and role names are checked one by one, and IDs and hint wording are
  filled in by the program. If validation fails, the reasons go back to the model for one rewrite; if that fails too, an error
  is shown and the ledger is left untouched.
- With the "Simulated" source the model may only use person, bag and table (all the simulated picture can draw), so a
  generated list can be shot to the end in simulation as well.
- Changing the list clears progress. "Start over" clears progress only: the list stays and the model is not asked again.
  "Back to the example list" returns to the three hand-written shots.
- To use the organisers' model on site, set three environment variables: `SHOTAGENT_LLM_BASE_URL`, `SHOTAGENT_LLM_API_KEY`,
  `SHOTAGENT_LLM_MODEL` (it has to be an OpenAI-compatible chat/completions API).
- To walk this path with no key: `python scripts/fake_endpoints.py`, then
  `OPENAI_BASE_URL=http://127.0.0.1:9001/v1 OPENAI_API_KEY=fake python run.py`. A hard-coded list comes back.

## How the modules relate

In one sentence: **both loops share one ledger, but only the planner and the slow loop can write it**. The fast loop takes a
snapshot when each shot starts and only reads from then on; `session.py` is the state machine that ties them together, and it
does not write the ledger either.

### Who calls whom

```
  web/  (browser)                                                         buttons: roll / end take / reshoot / start over
  sources.js --JPEG frame + timestamp + capabilities--> server.py  <--------------+
                                                            |
                                                     frame_source.py      -> Frame
                                                            |
                                                       session.py         state machine: framing -> recording -> checking
            +---------------------+-------------------------+-------+-------------------------+
            | every frame         | each frame of a take            | take finished           | boot / start over / new script
            v                     v                                 v                         v
      fast_loop.py            takes.py                        slow_loop.py               planner.py
       |        |         (collect, save)                      |        |                  |        |
  perception/  constraints.py                               judge/   constraints.py    plans.py  plan_gen.py
   yolo        spatial.py                                    mock    spatial.py        (example  (idea / script -> list;
   synthetic                                                 cosmos                     list)     llm.py asks the model)
   null
            :                                                       |                         |
            : snapshot only (once, when a shot starts)              | commit()                | commit()
            :                                                       v                         |
            +. . . . . . . . . . . . . . . . . . . . . . . . .>  ledger/  <-------------------+
                                                                 store.py    the single write path
                                                                 reducer.py  rules: kept take, which source a field trusts
                                                                 state.py    shot log + scene state
```

Solid lines are calls, the dotted line is "snapshot only". Both loops use `constraints.py` / `spatial.py`: the fast loop to
judge each frame, the slow loop to aggregate a take's YOLO observations into scene facts with the same geometry, so what is
written to the ledger and what is read back are measured with the same ruler.

### What each file is for

| File | Job | Reads ledger | Writes ledger |
|---|---|---|---|
| `contracts.py` | Data structures passed between modules (frame, detection, constraint, hint, take, verdict) | | |
| `config.py`, `texts.py`, `media.py` | Tunable parameters; shared wording; image and video encoding | | |
| `frame_source.py` + `web/js/sources.js` | Video source → one image + timestamp + capabilities | | |
| `planner.py` + `plans.py` | Request → shot list: the three hand-written shots for an empty request, the generator otherwise | yes | shot list |
| `plan_gen.py` + `llm.py` | The generator: idea / script → list (menu, validation, retry); an OpenAI-compatible model client | | |
| `fast_loop.py` | One frame → a framing hint | snapshot only | cannot |
| `constraints.py` + `spatial.py` | Relation predicates; geometry for relative position and facing | | |
| `perception/` | Detectors: YOLO / simulated picture / null | | |
| `slow_loop.py` | One take → verdicts + scene facts | yes | take settlement, scene facts |
| `judge/` | Semantic judges: mock / Cosmos | | |
| `takes.py` | Collects a take's frames and saves them | | |
| `ledger/` | The ledger: state, events, rules, the write path | | |
| `session.py` | The agent loop's state machine, tying the above together | read-only functions | cannot (it can only ask the planner to reload or the slow loop to settle) |
| `wiring.py` | Creates and connects every object | | |
| `server.py` | HTTP endpoints, a thin shell | | |

### What is passed between modules

Apart from the ledger's own two (events, state), everything is defined in `contracts.py`.

| Data | Produced by | Used by |
|---|---|---|
| `SourceCaps`, what a video source can do | the browser registering (`/api/source`) | `fast_loop.load`: decides which constraints can be checked and which degrade |
| `Frame`, one frame | `frame_source.ingest` | `session.on_frame` → `fast_loop.step` |
| `Observation`, what was seen in a frame | `fast_loop.step` (detector output + sensor readings) | returned with `Guidance`; stored in the take while recording; the slow loop aggregates positions from it |
| `Guidance`, a framing hint + per-constraint results | `fast_loop.step` | the UI; while recording, `session` adds it to the take together with the frame |
| `Take`, a take saved to disk | `takes.TakeStore.save` | `slow_loop.settle` |
| `JudgeRequest` → `JudgeVerdict` | `slow_loop` asks → `judge/` answers | back in `slow_loop` |
| `Plan`, a shot list | `plans.py` (hand-written) or `plan_gen.PlanGenerator` (written by a model, validated by the program) | `planner` wraps it in `PlanLoaded` and writes it to the ledger |
| Events (`ledger/events.py`) | `planner`: `PlanLoaded`; `slow_loop`: `TakeSettled`, `SceneFactObserved` | `Ledger.commit` → `reducer.apply` |
| `LedgerState` (`ledger/state.py`) | `Ledger.snapshot()`, or the return value of `commit()` | `session` (finds the current shot, feeds the UI), `fast_loop.load` (the snapshot), `slow_loop` |
| `Settlement`, the outcome of settling a take | `slow_loop.settle` | `session`: next shot on a pass, reshoot in place on a fail |

### Who may import whom

Each row imports only rows above it (within a row, `A <- B` means B imports A). Nothing goes the other way and there are no cycles:

```
foundation     contracts  config  texts  media  llm            import no internal module
geometry       spatial
ledger         ledger/state  <-  events  <-  reducer  <-  store   readers touch state only
parts          plans  perception/*  judge/*  takes  frame_source
predicates     constraints                                     uses spatial and ledger/state
generator      plan_gen      uses constraints (only for the PREDICATES table)
loops          fast_loop     uses constraints, perception/base, ledger/state (no write side)
               slow_loop     uses constraints, judge/base, ledger/state + events + store
               planner       uses plans, plan_gen, ledger/events + store
state machine  session       uses fast_loop, slow_loop, planner, takes, ledger/state
assembly       wiring  <-  server  <-  run.py
```

This table is executable: `tests/test_architecture.py` states, module by module, who may import whom, and checks it against
the real code on every test run. "The fast loop never writes the ledger" holds by structure: it does not import the ledger's
write side, and `wiring.py` never hands it the `Ledger` object.

### The journey of a frame (fast loop)

1. `sources.js` takes a frame → `POST /api/frame` (the next frame is only sent after the reply, so the server always works on the latest picture)
2. `frame_source.ingest` decodes it into a `Frame`
3. `session.on_frame`: while framing or recording → hands it to `fast_loop.step`
4. `fast_loop.step`: the detector produces an `Observation` → `constraints.evaluate` for each constraint → the one line most worth saying is picked
5. While recording, `session` adds "this frame + the fast loop's result for it" to the current take
6. The reply carries the hint, the detection boxes and anything recovered from the ledger; the UI draws them

### The journey of a take (slow loop)

1. Roll: `session` creates a `TakeBuffer` and starts a timer (a fixed 5 seconds; the take can be ended earlier by hand)
2. End: `session` enters "checking" and runs `takes.save` → `slow_loop.settle` in the background
3. `settle`: aggregates the per-frame fast-constraint results → asks `judge` only if they all pass → on a pass, aggregates scene
   facts from the per-frame detections → one `ledger.commit([...], writer="slow_loop")`
4. `ledger/reducer.py` applies the rules, `store.py` writes to disk, the version goes up by one
5. `session` returns to framing: reads the ledger again, works out the current shot, hands the fast loop a new snapshot

### The ledger

- Files: `data/ledger.json` (current state) + `data/ledger.events.jsonl` (the full record of every write, append-only)
- Two layers: `shots` (the shot log) and `scene` (the scene state). Every constraint verdict and every scene fact carries its
  source: `plan` / `yolo` / `cosmos` / `sensor`
- `Ledger.commit()` is the only way to write; who may commit what is fixed in `store.WRITE_PERMISSIONS`
- All rules are in `ledger/reducer.py`: one kept take per shot (a new take replaces the old one only if it passes; if neither
  passes the latest is kept); position trusts YOLO, description trusts Cosmos; scene facts may only come from a kept take that
  passed, and when a take is replaced the facts it brought are dropped with it
- Files of replaced takes stay in `data/takes/`; they are just no longer in the index

## Decisions the design doc did not make

Some things were left open by the design doc, or two passages did not agree. These are the choices made; all can be changed:

1. **The fast loop never writes the ledger, yet position fields must come from YOLO.** The fast loop hands what it saw in each
   frame out with the hint, and it is collected in the take while recording; the slow loop aggregates it at settlement and
   writes it with `source=yolo`. (`slow_loop._position_fact`)
2. **When a take counts as passed.** Each constraint in the table has its own checker, so: fast constraints are aggregated as
   "share of satisfying frames ≥ 60%", slow constraints are answered by Cosmos, and both must pass. If a fast constraint
   fails, Cosmos is not asked. (`slow_loop.settle`, `Tuning.pass_ratio`)
3. **An extra `session.py`.** It is not among the suggested modules, but someone has to decide which shot is current, who
   gets a frame, and when to settle. The current shot is not stored; it is derived from the ledger each time.
4. **"Load the snapshot once when shooting starts" is read as once per shot, when it starts framing.** Loaded once for the whole
   session, shot two would never see the fact shot one just wrote, and demo moment two would not work.
5. **Degrading is one uniform mechanism.** Capabilities = the detector's capabilities ∪ the video source's sensors; a
   constraint whose needs are not met becomes a text prompt that says what is missing. No YOLO, no pose model and no gyro all
   take the same path. Degraded constraints do not count toward a pass, and the ledger records them as unchecked, never as passed.
6. **Hidden subjects are recovered from the ledger.** When the bag is hidden, if the ledger holds its position relative to the
   table and the table is still visible, the fast loop projects it into the current picture (a dashed box in the UI). Live
   detection wins; when the two disagree on left / right the UI says so and follows the picture.
7. **Distance is measured horizontally only.** With the person seated and the bag under the table, a height difference does
   not mean far apart. (`spatial.gap_ratio`)
8. **The three shots were chosen here.** S01 produces a scene fact, S02 is the constraint table from the design doc, and S03's
   pitch constraint is there to exercise the sensor fallback.
9. **One HTTP request per frame, no WebSocket.** "Process only the latest frame" comes for free, and every endpoint can be
   tried by itself with curl.
10. **The fast loop's YOLO runs locally.** The organisers' YOLO endpoint takes video clips and does not suit per-frame use (see
    the next section).
11. **Semantic constraints carry a separate `ask`.** `text` is the statement shown to people; `ask` is the yes/no question
    put to Cosmos.
12. **A "Simulated" source was added.** It is not in the design doc; it is there so the loop can be verified without a camera
    or a model.
13. **Generating a list is one call plus validation, not a multi-turn agent.** Design doc 2.2 describes "style cards → list";
    here it is "idea or script → list", with no style-card layer. The model only fills in what to film and what to check; IDs,
    checkers and hint wording are filled in by the program from each predicate's spec.
14. **The roles and predicates offered to the model are a menu computed from the code.** Register a new predicate or add a
    prop to the label map, and the next generation can use it with no prompt to edit.
15. **Only one of each role is tracked.** So the prompt asks for stories with at most one person on screen; scenes with
    several people cannot be written yet.
16. **The fast loop keeps only the roles in the current list.** A dozen props (chair, cup...) were added to the label map;
    detections the current list does not mention are not passed on.

## Connecting the real models

**YOLO**: used by default once `pip install ultralytics` is done (`yolo11n` + `yolo11n-pose`; the weights download into the
current directory on first run, so run it once on your own network before the event instead of at the venue).
Run `python scripts/check_yolo.py` first with the bag, table and person set up as in the demo, and see how stable recognition
is. It lists every class recognised in the picture; if your bag is recognised as something else, add a line to
`config.DEFAULT_LABEL_MAP`.

**Cosmos**: `COSMOS3_REASON_URL=... python run.py --judge cosmos`. The variable names match the organisers' VM.
The first time, run `python scripts/probe_endpoints.py` to see what the model replies and how long a take takes round trip.
Until you have the real endpoint, `python scripts/fake_endpoints.py` starts a fake one with the same API shape to practise on.

### The organisers' environment (from the starter repo relaxedtomato/vast-builders-challenge, version of 2026-09-30)

| Question in the design doc | What the starter repo says | What it means for this skeleton |
|---|---|---|
| How to call Cosmos | `$COSMOS3_REASON_URL/v1/chat/completions`, OpenAI-compatible, model `nvidia/cosmos3-reason`, video embedded as `data:video/mp4;base64,...`, no auth | `judge/cosmos.py` is written to exactly this. Note: the build-day guide says these models are "not called directly, you query what they generated", while the gpu folder documents how to call them directly. Whether direct calls are allowed, and any rate limit, must be confirmed on site |
| How to call YOLO | `$YOLO_URL/v1/infer`, body `video_base64` (a whole clip), a custom service with `/openapi.json` | The per-frame fast loop uses local YOLO; save the remote reply's structure with `probe_endpoints.py` before deciding whether to use it |
| Embeddings | `$COSMOS_EMBED1_URL/v1/embeddings`, 256 dimensions | For semantic search later, on the ledger's description fields |
| Where to develop | A remote VM in the browser, with credentials and endpoints in the VM's environment; only web apps, CLIs or scripts | The camera is on your laptop, so where the server runs has to be decided on site, see below |
| Uploading your own video | The ingest notes say only re-ingest of existing footage is supported this time and direct upload is not a supported workflow (the repo still contains an upload-video skill; the two disagree) | The ledger is its own index and does not depend on the organisers' ingest; getting your own footage into their search has to be asked on site |
| An LLM for your own logic | Weights & Biases serverless inference, keys in the environment | The shot-list generator can point at it through the three `SHOTAGENT_LLM_*` variables |

**Where the server runs** (an inference; the starter repo does not say): browsers only expose the camera over `https` or on `localhost`.

- Server on the laptop (`localhost`, camera works): the Cosmos endpoint must be reachable from the laptop. The first thing to do
  on site is to run `probe_endpoints.py` on the laptop.
- If it is not reachable, the server has to run on the VM: the laptop's browser then needs an `https` address for it (a tunnel,
  or an entry point from the organisers), frames travel over the network to the VM, and the fast loop gains one round trip of latency.

## Not done yet, and known issues

Not verified:
- Shot-list generation has never been run against the real OpenAI. The request format, validation and retry were tested against
  a fake server, but the quality of what `gpt-5.6-luna` actually writes, how long it takes, and whether it accepts this kind
  of request are only known once you run `scripts/try_plan.py` with a key. The model name and price came from third-party
  pricing pages and were not confirmed on OpenAI's own page
- Whether the organisers' Weights & Biases inference service is OpenAI-compatible, and which models it offers
- How reliably YOLO recognises the newly added props (chair, cup, book...): they are COCO classes, nothing more was tested
- The real Cosmos endpoint (request format and parsing were tested against a fake server)
- Whether YOLO recognises your bag and table, and bags are a weak spot. Measured on COCO128 (128 labelled sample images, which
  are part of the model's own training data): of the 14 images labelled with a bag, `yolo11n` found a bag in only 6 and
  `yolo11s` in 9; of the 10 labelled with a dining table, both models found it in 8. So the design doc's fallback of using
  props with strong colours is worth preparing in advance, or try `--yolo-model yolo11s.pt` straight away
- The person's facing was only tuned on photos: the thresholds were set on some thirty COCO128 photos with people, where
  frontal, profile and back views are mostly told apart correctly; in-between cases (head down, half turned) count as "facing
  the camera". A real person turning their head in front of a camera has not been tried; check with `check_yolo.py`
- A real camera (the browser side was tested with Chromium's fake camera device)
- The tests ran on Linux with Python 3.10 to 3.13 (3.10 to 3.12 without ultralytics, so the 4 YOLO tests were skipped there);
  the only browser tried is Chromium. macOS / Windows and Safari / Firefox were not tried

Left out of the skeleton on purpose:
- "Please confirm" is only a flag (low-confidence scene facts are marked in the UI); there is no confirm action
- With several detections of one kind the most confident is used (scenes with several people or bags will pick wrong)
- A generated list has no "look before you accept" step: it replaces the current list as soon as it is written (after one
  confirmation, since progress is cleared)
- Hints are not smoothed over time, so they jump when detection jitters
- A take is the uploaded sampled frames, not full-frame-rate video (the top of `takes.py` says how to switch to a MediaRecorder upload)
- Left and right in the ledger follow the picture, so they flip if the camera moves to the other side of the table
- Reshooting an earlier shot that changes the scene state does not flag later shots that already passed for review
- All wording is English. To translate it: shot-specific text is in `plans.py`, shared wording in `texts.py`, and sentences
  only one module says stay in that module (the list is at the top of `texts.py`); the front end is `web/index.html` and the
  files under `web/js/`

Where the design doc's deferred items plug in:

| Deferred item | Where it goes |
|---|---|
| Detecting the start and end of a take automatically | `session.py`: call `start_take` / `stop_take` from `on_frame` based on `guidance.ready`; keep the manual button |
| Style cards | Idea / script → list is done (`plan_gen.py`). Style cards can be added to `plan_gen.system_prompt` as reference material |
| Having the model ask a few questions before settling on a list | `session.replan` is one question, one answer today; for several turns add the dialogue inside `plan_gen.PlanGenerator`, the output is still a `Plan` |
| Semantic search | Read each take's `description` from `ledger.snapshot()`; embed with Embed1 |
| Automatic rough cut | Take each shot's `effective_take.clip_dir` in `shots` order and join them with ffmpeg |
| Full conflict arbitration and an edit trail | Add events and rules in `ledger/reducer.py`; the write log is already append-only |
| GO Ultra (RTMP) | Write a process that pulls the stream, samples frames and calls `/api/source` and `/api/frame` like the browser |
| Occlusion constraints | Add a predicate in `constraints.py` and register it in `PREDICATES` |

## Layout

```
run.py                      entry point
shotagent/                  backend (start with the notes in __init__.py)
web/                        UI: index.html, style.css, js/{main,api,sources,render}.js
scripts/simulate.py         narrated run of the whole flow in the terminal
scripts/check_yolo.py       does YOLO recognise your bag, table and person
scripts/probe_endpoints.py  probes the organisers' Cosmos / YOLO / Embed endpoints
scripts/fake_endpoints.py   fake endpoints with the same API shape (can also stand in for the shot-list model)
scripts/try_plan.py         try once by itself: idea / script -> shot list
tests/                      test_architecture (dependency rules), test_ledger, test_spatial_constraints,
                            test_fast_loop, test_slow_loop, test_flow_http (the whole loop),
                            test_judge_cosmos, test_synthetic_media, test_yolo_optional,
                            test_plan_gen (generator and model client), test_plan_flow (switching lists end to end)
data/                       created at run time: ledger.json, ledger.events.jsonl, takes/
```
