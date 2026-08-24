/* Heart Rate — heartbeat ECG analyzer
 *
 * Every number on this page is real. The waveforms are ECG recordings from
 * the MIT-BIH Arrhythmia Database, the verdicts come from the trained model
 * in model.pt, and the "actually abnormal" counts are what the cardiologists
 * wrote down in the 1970s. Nothing here is simulated.
 *
 * The heavy lifting happened in precompute.py — the browser just draws the
 * results it wrote into site_data/.
 */

const DATA = "site_data";
const SECONDS_ON_SCREEN = 6;

const $ = (id) => document.getElementById(id);
const canvas = $("trace");
const ctx = canvas.getContext("2d");

let index = null;      // the list of available records
let current = null;    // the record being shown: beats, counts, scores
let wave = null;       // its waveform, as an Int16Array

/* ---------- loading ---------- */

async function loadIndex() {
  index = await (await fetch(`${DATA}/index.json`)).json();
  const select = $("record-select");
  select.innerHTML = '<option value="">Select a record…</option>';
  for (const rec of index.records) {
    const option = document.createElement("option");
    option.value = rec.record;
    option.textContent =
      `Record ${rec.record} — ${rec.abnormalTrue.toLocaleString()} abnormal beats, ` +
      `${Math.round(rec.recall * 100)}% caught`;
    select.appendChild(option);
  }
}

async function loadRecord(name) {
  const [info, buffer] = await Promise.all([
    fetch(`${DATA}/${name}.json`).then((r) => r.json()),
    fetch(`${DATA}/${name}.bin`).then((r) => r.arrayBuffer()),
  ]);
  current = info;
  wave = new Int16Array(buffer);
}

/* ---------- what happened to each beat ---------- */

// Because we have the cardiologists' labels as well as the model's guess, we
// can show not just what the model claimed but whether it was right. That
// honesty is the point of the page.
function verdict(beat) {
  const modelSaysAbnormal = beat.said !== 0;
  const reallyAbnormal = beat.true !== 0;
  if (modelSaysAbnormal && reallyAbnormal) return "caught";
  if (!modelSaysAbnormal && reallyAbnormal) return "missed";
  if (modelSaysAbnormal && !reallyAbnormal) return "false";
  return "normal";
}

const COLOR = {
  caught: "#E03131",     // flagged, and it really was abnormal
  missed: "#F59F00",     // abnormal, and the model walked straight past it
  false: "#4C6EF5",      // flagged, but the beat was fine
  normal: null,          // nothing drawn
};

/* ---------- drawing ---------- */

function css(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function windowSize() {
  return Math.round(SECONDS_ON_SCREEN * current.hz);
}

function draw(startSample) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const span = windowSize();
  const end = Math.min(startSample + span, wave.length);
  const pad = 26;
  const usable = height - pad * 2;

  // Scale the visible slice to fill the box. The signal's range varies a lot
  // between patients and even within one recording, so fitting to what is
  // actually on screen beats a fixed scale.
  let low = Infinity, high = -Infinity;
  for (let i = startSample; i < end; i++) {
    if (wave[i] < low) low = wave[i];
    if (wave[i] > high) high = wave[i];
  }
  const range = Math.max(high - low, 1);
  const yAt = (v) => pad + (1 - (v - low) / range) * usable;
  const xAt = (i) => ((i - startSample) / span) * width;

  // Highlight blocks behind flagged beats, drawn first so the trace sits on top.
  for (const beat of current.beats) {
    if (beat.at < startSample || beat.at >= end) continue;
    const kind = verdict(beat);
    if (kind === "normal") continue;
    const x = xAt(beat.at);
    const w = Math.max((width / span) * current.hz * 0.22, 10);
    ctx.fillStyle = COLOR[kind] + "33";
    ctx.beginPath();
    ctx.roundRect(x - w / 2, 6, w, height - 12, 5);
    ctx.fill();
  }

  ctx.strokeStyle = css("--text-dim");
  ctx.lineWidth = 1.3;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = startSample; i < end; i++) {
    const x = xAt(i), y = yAt(wave[i]);
    if (i === startSample) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // A letter above each flagged beat saying what the model called it.
  ctx.font = "500 11px system-ui, sans-serif";
  ctx.textAlign = "center";
  for (const beat of current.beats) {
    if (beat.at < startSample || beat.at >= end) continue;
    const kind = verdict(beat);
    if (kind === "normal") continue;
    const label = kind === "missed" ? index.classes[beat.true]
      : index.classes[beat.said];
    ctx.fillStyle = COLOR[kind];
    ctx.fillText(label, xAt(beat.at), 17);
  }
}

/* ---------- the summary, written from the real numbers ---------- */

function describe(info) {
  const rate = info.abnormalTrue / info.total;
  const perMinute = info.abnormalTrue / info.minutes;

  // Where in the recording do the abnormal beats fall? Comparing the first
  // half against the second is enough to tell "steady" from "worsening".
  const half = info.samples / 2;
  const abnormal = info.beats.filter((b) => b.true !== 0);
  const late = abnormal.filter((b) => b.at >= half).length;
  const share = abnormal.length ? late / abnormal.length : 0.5;

  let howMany;
  if (rate < 0.005) {
    howMany = "almost entirely steady, with only a stray abnormal beat";
  } else if (rate < 0.05) {
    howMany = `mostly steady — about ${Math.round(rate * 1000) / 10}% of beats are abnormal`;
  } else if (rate < 0.2) {
    howMany = `frequent abnormal beats, roughly 1 in every ${Math.round(1 / rate)}`;
  } else {
    howMany = `heavily irregular — about ${Math.round(rate * 100)}% of all beats are abnormal`;
  }

  let whereAbout;
  if (abnormal.length < 10) {
    whereAbout = "";
  } else if (share > 0.68) {
    whereAbout = " They cluster toward the end of the recording rather than spreading evenly across it.";
  } else if (share < 0.32) {
    whereAbout = " They are concentrated early on and settle down as the recording goes.";
  } else {
    whereAbout = " They are spread fairly evenly across the half hour.";
  }

  const perMin = abnormal.length >= 10
    ? ` That works out to about ${Math.round(perMinute)} a minute.` : "";

  return `Over ${info.minutes} minutes this recording is ${howMany}.${whereAbout}${perMin}`;
}

function describeModel(info) {
  const recall = info.abnormalTrue ? info.caught / info.abnormalTrue : 1;
  const parts = [
    `The model found ${info.caught.toLocaleString()} of the ` +
    `${info.abnormalTrue.toLocaleString()} beats the cardiologists marked abnormal`,
  ];
  if (info.missed) parts.push(`missed ${info.missed.toLocaleString()}`);
  if (info.falseAlarm) {
    parts.push(`and flagged ${info.falseAlarm.toLocaleString()} healthy beats it should have left alone`);
  }
  let text = parts.join(", ") + ".";
  if (recall < 0.5) {
    text += " This is one of the patients it handles badly.";
  } else if (recall > 0.95 && info.falseAlarm < info.caught * 0.3) {
    text += " A strong result on this patient.";
  }
  return text;
}

/* ---------- wiring ---------- */

function clockAt(sample) {
  const seconds = Math.round(sample / current.hz);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function startFromSlider() {
  const fraction = Number($("offset").value) / 1000;
  return Math.floor(fraction * Math.max(0, wave.length - windowSize()));
}

function refreshTrace() {
  const start = startFromSlider();
  draw(start);
  $("offset-out").textContent = clockAt(start);
}

async function analyze() {
  const name = $("record-select").value;
  const hint = $("picker-hint");
  if (!name) {
    hint.textContent = "Pick a recording first.";
    hint.style.color = "var(--red-dim)";
    return;
  }

  hint.textContent = "Analyzing…";
  hint.style.color = "";
  await loadRecord(name);

  hint.textContent = `Record ${name}: ${current.minutes} minutes, ` +
    `${current.total.toLocaleString()} beats, ${current.hz} Hz. The model and the ` +
    `cardiologists agree on ${(current.agreement * 100).toFixed(1)}% of them.`;

  $("record-badge").textContent = `record ${name}.dat`;
  $("stat-total").textContent = current.total.toLocaleString();
  $("stat-flagged").textContent = current.abnormalSaid.toLocaleString();
  $("stat-truth").textContent = current.abnormalTrue.toLocaleString();
  $("stat-caught").textContent = current.abnormalTrue
    ? `${Math.round((current.caught / current.abnormalTrue) * 100)}%` : "—";

  $("summary").textContent = describe(current);
  $("summary-note").textContent = describeModel(current);

  $("offset").value = 0;
  ["trace-panel", "stats", "summary-panel"].forEach((s) => ($(s).hidden = false));
  refreshTrace();
}

$("analyze-btn").addEventListener("click", analyze);
$("offset").addEventListener("input", () => { if (wave) refreshTrace(); });
window.addEventListener("resize", () => { if (wave) refreshTrace(); });

loadIndex().catch(() => {
  $("picker-hint").textContent =
    "Couldn't load the data. This page needs to be served over http, not opened as a file.";
  $("picker-hint").style.color = "var(--red-dim)";
});
