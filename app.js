/* Heart Rate — heartbeat ECG analyzer (layout stage)
 *
 * Everything below is DEMO data. When the real model is trained, the only
 * thing that changes is where `beats` and `stats` come from: instead of
 * being made up here, they'll be fetched from the Python side.
 */

const RECORDS = {
  "100": {
    total: 2273, abnormalRate: 0.01, confidence: 99,
    summary: "This recording is steady almost the whole way through. A handful of early beats turn up, but they're scattered and infrequent — nothing that forms a pattern."
  },
  "208": {
    total: 2955, abnormalRate: 0.19, confidence: 98,
    summary: "This recording shows a mostly steady rhythm with frequent early beats — about 1 in every 5. They cluster toward the end of the recording rather than spreading evenly across it."
  },
  "203": {
    total: 2980, abnormalRate: 0.31, confidence: 94,
    summary: "The rhythm here is irregular throughout, with abnormal beats appearing in bursts rather than isolated. The signal is also noisier than most, so these numbers are less certain."
  }
};

const BEATS_ON_SCREEN = 8;
const RECORDING_SECONDS = 30 * 60;

let beats = [];          // one entry per beat: { abnormal: true/false }
let current = null;      // the record we're showing

/* ---------- elements ---------- */

const $ = (id) => document.getElementById(id);
const canvas = $("trace");
const ctx = canvas.getContext("2d");

/* ---------- demo data ---------- */

// A tiny predictable random generator, so the same record always looks
// the same instead of reshuffling every time you move the slider.
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
}

function makeBeats(record, seed) {
  const rand = seededRandom(seed);
  const out = [];
  for (let i = 0; i < record.total; i++) {
    // Later in the recording, abnormal beats get more likely — this is
    // what makes the "they cluster toward the end" summary line true.
    const position = i / record.total;
    const chance = record.abnormalRate * (0.4 + 1.2 * position);
    out.push({ abnormal: rand() < chance });
  }
  return out;
}

/* ---------- drawing ---------- */

function css(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

// One heartbeat, drawn as a list of [x, y] points across a 0..1 width.
// A normal beat has a small bump, a tall spike, then a rounded wave.
// An abnormal beat is wider and lurches the other way — that's roughly
// what a premature ventricular beat looks like on a real strip.
function beatShape(abnormal) {
  return abnormal
    ? [[0,.5],[.18,.5],[.28,.30],[.42,1.0],[.55,.05],[.68,.62],[.82,.5],[1,.5]]
    : [[0,.5],[.14,.5],[.20,.43],[.26,.5],[.42,.5],[.46,.44],[.50,.10],[.54,.86],[.58,.5],[.72,.5],[.80,.38],[.88,.5],[1,.5]];
}

function draw(startBeat) {
  // Match the canvas to its on-screen size so the line isn't blurry.
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const slice = beats.slice(startBeat, startBeat + BEATS_ON_SCREEN);
  if (!slice.length) return;

  const beatWidth = width / BEATS_ON_SCREEN;
  const top = 20;
  const usable = height - top * 2;

  // Highlight blocks behind the abnormal beats.
  ctx.fillStyle = "rgba(224, 49, 49, 0.20)";
  slice.forEach((beat, i) => {
    if (!beat.abnormal) return;
    const x = i * beatWidth + beatWidth * 0.12;
    ctx.beginPath();
    ctx.roundRect(x, 6, beatWidth * 0.76, height - 12, 6);
    ctx.fill();
  });

  // The trace itself, drawn as one continuous line.
  ctx.strokeStyle = css("--text-dim");
  ctx.lineWidth = 1.6;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  slice.forEach((beat, i) => {
    beatShape(beat.abnormal).forEach(([px, py], j) => {
      const x = (i + px) * beatWidth;
      const y = top + (1 - py) * usable;
      if (i === 0 && j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
  });
  ctx.stroke();
}

/* ---------- wiring ---------- */

function clockAt(beatIndex) {
  const seconds = Math.round((beatIndex / beats.length) * RECORDING_SECONDS);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function startBeatFromSlider() {
  const fraction = Number($("offset").value) / 100;
  return Math.floor(fraction * Math.max(0, beats.length - BEATS_ON_SCREEN));
}

function refreshTrace() {
  const start = startBeatFromSlider();
  draw(start);
  $("offset-out").textContent = clockAt(start);
}

function analyze() {
  const id = $("record-select").value;
  if (!id) {
    $("picker-hint").textContent = "Pick a recording first.";
    $("picker-hint").style.color = "var(--red-dim)";
    return;
  }

  $("picker-hint").textContent = "Demo data for now — the real model isn't wired up yet.";
  $("picker-hint").style.color = "";

  current = RECORDS[id];
  beats = makeBeats(current, Number(id) * 7919);

  const abnormal = beats.filter((b) => b.abnormal).length;

  $("record-badge").textContent = `record ${id}.dat`;
  $("stat-total").textContent = current.total.toLocaleString();
  $("stat-abnormal").textContent = abnormal.toLocaleString();
  $("stat-confidence").textContent = `${current.confidence}%`;
  $("summary").textContent = current.summary;

  $("offset").value = 0;
  ["trace-panel", "stats", "summary-panel"].forEach((s) => ($(s).hidden = false));
  refreshTrace();
}

$("analyze-btn").addEventListener("click", analyze);
$("offset").addEventListener("input", refreshTrace);
window.addEventListener("resize", () => { if (beats.length) refreshTrace(); });
