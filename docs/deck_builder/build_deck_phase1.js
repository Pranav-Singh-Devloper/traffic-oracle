// Traffic Oracle — Team T042 presentation. Usage: node build_deck.js <repo_root> <out.pptx>
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = process.argv[2];
const OUT = process.argv[3];
const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "data.json"), "utf8"));
const SZ = JSON.parse(fs.readFileSync(path.join(HERE, "figsizes.json"), "utf8"));
const FIG = (f) => path.join(ROOT, "results/figures", f + ".png");

const THEME = {
  name: "Traffic Oracle Night", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "0A0F1E", lt1: "FFFFFF", dk2: "151C35", lt2: "AEB8D0", accent1: "FF5A4E", accent2: "FFB547",
    accent3: "3DD6C4", accent4: "8C9BFF", accent5: "6B7794", accent6: "5BD68A", hlink: "FFB547", folHlink: "8C9BFF" },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Traffic Oracle — Phase 1 — Team T042";
pres.author = "Team T042";
const C = pres.SchemeColor;
const WHITE = C.background1, MUTED = C.background2, CARD = C.text2;
const RED = C.accent1, AMBER = C.accent2, TEAL = C.accent3, INDIGO = C.accent4, SLATE = C.accent5, GREEN = C.accent6;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE", background: { path: path.join(HERE, "bg_title.jpg") },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.0, w: 11.7, h: 1.5, fontSize: 66, bold: true, color: WHITE, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 2.6, w: 10.5, h: 0.9, fontSize: 26, color: MUTED, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION", background: { path: path.join(HERE, "bg_section.jpg") },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.8, y: 1.9, w: 8, h: 0.5, fontSize: 16, bold: true, color: AMBER, charSpacing: 4, align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.4, w: 11.7, h: 1.4, fontSize: 50, bold: true, color: WHITE, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.85, w: 10.5, h: 1.2, fontSize: 22, color: MUTED, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { path: path.join(HERE, "bg_content.jpg") },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.38, w: 12.1, h: 0.85, fontSize: 36, bold: true, color: WHITE, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "TRAFFIC ORACLE  ·  TEAM T042", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: SLATE, charSpacing: 3, margin: 0 } } },
  ],
  slideNumber: { x: 12.25, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: SLATE, align: "right" },
});

// ---------------------------------------------------------------- helpers
const shadow = () => ({ type: "outer", color: "000000", blur: 14, offset: 3, angle: 90, opacity: 0.45 });
function card(s, x, y, w, h, fill = CARD, transparency = 8) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill, transparency }, line: { color: fill, transparency: 100 }, rectRadius: 0.12, shadow: shadow(), objectName: "card" });
}
function T(s, text, o) { s.addText(text, { isTextBox: true, margin: 0, color: WHITE, fontSize: 16, valign: "top", ...o }); }
function stat(s, x, y, w, h, big, label, color = AMBER, bigSize = 46) {
  card(s, x, y, w, h);
  T(s, big, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: h * 0.5, fontSize: bigSize, bold: true, color, fontFace: "Arial", valign: "middle" });
  T(s, label, { x: x + 0.3, y: y + 0.25 + h * 0.5, w: w - 0.6, h: h * 0.5 - 0.35, fontSize: 14, color: MUTED });
}
function bullets(s, items, x, y, w, h, size = 16, color = WHITE) {
  const runs = items.map((it, i) => {
    const base = { bullet: { indent: 18 }, paraSpaceAfter: 8, breakLine: i < items.length - 1, color, fontSize: size };
    return typeof it === "string" ? { text: it, options: base } : { text: it.t, options: { ...base, ...it.o } };
  });
  s.addText(runs, { isTextBox: true, x, y, w, h, margin: 0, valign: "top" });
}
function fig(s, name, x, y, w, h, pad = 0.12) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: WHITE }, line: { color: WHITE, transparency: 100 }, rectRadius: 0.1, shadow: shadow(), objectName: "figure frame" });
  const [pw, ph] = SZ[name];
  const ar = pw / ph, bw = w - 2 * pad, bh = h - 2 * pad;
  let iw = bw, ih = bw / ar;
  if (ih > bh) { ih = bh; iw = bh * ar; }
  s.addImage({ path: FIG(name), x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, altText: name.replace(/[_/]/g, " ") });
}
function chartBase(extra) {
  return {
    catAxisLabelColor: HEX.lt2, valAxisLabelColor: HEX.lt2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, valGridLine: { color: "26304F", size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: false, valAxisLineShow: false, legendColor: HEX.lt2, legendFontFace: "+mn-lt", legendFontSize: 12,
    dataLabelColor: "FFFFFF", dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, titleColor: "FFFFFF", titleFontFace: "+mn-lt", titleFontSize: 14,
    valAxisTitleColor: HEX.lt2, catAxisTitleColor: HEX.lt2, valAxisTitleFontSize: 12, catAxisTitleFontSize: 12,
    valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt", ...extra,
  };
}
function arrow(s, x1, y1, x2, y2, color = HEX.lt2, width = 2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width, endArrowType: "triangle" } });
}
const mt = (m, h, k = 0) => D.mt[`${m}|${h}`][k];
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);

let section = "";
function slide(master, sectionTitle) {
  if (sectionTitle !== section) { pres.addSection({ title: sectionTitle }); section = sectionTitle; }
  return pres.addSlide({ masterName: master, sectionTitle });
}
function chapter(sec, kicker, title, body, notes) {
  const s = slide("SECTION", sec);
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  s.addText(body, { placeholder: "body" });
  s.addNotes(notes);
  return s;
}
const eda = D.eda;
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";

// ================================================================= 1. TITLE
{
  const s = slide("TITLE", "Opening");
  s.addText("Traffic Oracle", { placeholder: "title" });
  s.addText("Phase 1: how far can per-segment machine learning go?", { placeholder: "body" });
  T(s, [{ text: "Baselines, tuned trees and hand-made neighbour features on the METR-LA freeway network", options: { breakLine: true } },
        { text: "TEAM T042  ·  Pranav Singh · Kunal · Himanshu · Kanchan · Vaibhav", options: { color: AMBER, bold: true, charSpacing: 2, fontSize: 14 } }],
    { x: 0.8, y: 3.55, w: 11, h: 0.9, fontSize: 18, color: MUTED });
  s.addNotes("Open on the image: long-exposure traffic at night. Every streak is a car; every car is a data point. Our question is simple: when congestion starts somewhere on a freeway, can a model that looks at neighbouring sensors see it coming before a model that only looks at one road segment?");
}
// ================================================================= 2. COLD OPEN
{
  const s = slide("SECTION", "Opening");
  s.addText("LOS ANGELES · WEEKDAY, 8:00 AM", { placeholder: "kicker" });
  s.addText("207 sensors are listening", { placeholder: "title" });
  s.addText(`At rush hour, ${pct(eda.distribution.share_below_threshold_peak)} of readings fall below 40 mph — off-peak it is only ${pct(eda.distribution.share_below_threshold_offpeak)}. Forecasting is easy when traffic flows; the hard part is the jam.`, { placeholder: "body" });
  s.addNotes("Set the stakes. Congestion is the minority case overall but dominates errors: at weekday peaks more than a quarter of readings are below 40 mph. A forecaster is only useful if it handles the jam.");
}
// ================================================================= 3. QUESTION
{
  const s = slide("CONTENT", "Opening");
  s.addText("The question behind every slide", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.6, 5.2);
  T(s, "HYPOTHESIS", { x: 0.95, y: 1.85, w: 5, h: 0.35, fontSize: 14, bold: true, color: AMBER, charSpacing: 3 });
  T(s, "Congestion travels. A jam starts at a bottleneck and its tail moves upstream, so downstream sensors know an upstream sensor's future.", { x: 0.95, y: 2.3, w: 5, h: 1.6, fontSize: 18 });
  bullets(s, ["A per-segment model sees only its own history", "A model with neighbour features can read the roads around it",
    "Phase 1 tests how far hand-made neighbour features get us"], 0.95, 4.15, 5, 2.2, 16, MUTED);
  // shockwave diagram
  const y0 = 4.0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: y0, w: 5.9, h: 0.16, fill: { color: SLATE }, line: { color: SLATE, transparency: 100 }, rectRadius: 0.08 });
  const cols = [GREEN, GREEN, AMBER, RED, RED];
  ["A", "B", "C", "D", "E"].forEach((n, i) => {
    const x = 7.05 + i * 1.3;
    s.addShape(pres.shapes.OVAL, { x, y: y0 - 0.22, w: 0.6, h: 0.6, fill: { color: cols[i] }, line: { color: WHITE, width: 1.5 } });
    T(s, n, { x, y: y0 - 0.22, w: 0.6, h: 0.6, fontSize: 16, bold: true, color: "0A0F1E", align: "center", valign: "middle" });
  });
  arrow(s, 7.0, 2.85, 12.4, 2.85, HEX.accent2, 2.5);
  T(s, "Traffic flows downstream →", { x: 7.0, y: 2.25, w: 5.4, h: 0.45, fontSize: 16, color: AMBER, bold: true });
  arrow(s, 12.4, 5.2, 7.0, 5.2, HEX.accent1, 2.5);
  T(s, "← The congestion wave moves upstream", { x: 7.0, y: 5.35, w: 5.6, h: 0.45, fontSize: 16, color: RED, bold: true });
  T(s, "Sensors D and E slow down first; C, B and A follow minutes later.", { x: 7.0, y: 5.95, w: 5.6, h: 0.7, fontSize: 14, color: MUTED });
  s.addNotes("This is the physical intuition. Traffic moves forward, but the jam's tail moves backwards. If D and E slow down now, C will slow down soon. A model that sees D and E can warn C; a model that only sees C has to guess. Everything that follows tests whether that intuition survives contact with real data.");
}
// ================================================================= CHAPTER 1 DATA
chapter("01 The data", "CHAPTER 01", "Listening to 207 sensors", "Four months of Los Angeles freeway speeds — and what the silence in the data means.",
  "Chapter one: what we actually have, and the traps hidden in it.");
{
  const s = slide("CONTENT", "01 The data");
  s.addText("METR-LA in four numbers", { placeholder: "title" });
  const w = 2.85, g = 0.25;
  stat(s, 0.6, 1.6, w, 2.4, "207", "loop-detector sensors on LA County freeways", AMBER);
  stat(s, 0.6 + (w + g), 1.6, w, 2.4, "34,272", "5-minute timestamps · Mar 1 – Jun 27, 2012 (119 days)", TEAL);
  stat(s, 0.6 + 2 * (w + g), 1.6, w, 2.4, pct(eda.missingness.missing_rate_overall), "of readings are 0 mph — sensor outages, not stopped cars", RED);
  stat(s, 0.6 + 3 * (w + g), 1.6, w, 2.4, "1,515", "directed road links in the sensor graph", INDIGO);
  card(s, 0.6, 4.3, 12.1, 2.45);
  T(s, "WHAT THAT MEANS FOR MODELLING", { x: 0.95, y: 4.55, w: 11, h: 0.35, fontSize: 14, bold: true, color: AMBER, charSpacing: 3 });
  bullets(s, ["Speed only (mph) — no flow, no occupancy; maximum observed speed 70 mph",
    "Zeros are masked in every loss and metric and never used as targets",
    "Source: DCRNN benchmark (Li et al., ICLR 2018); files verified by SHA-256 on download"], 0.95, 5.0, 11.4, 1.7, 16);
  s.addNotes("Four numbers to remember. 207 sensors, 34 thousand five-minute steps, 8.1 percent missing, 1,515 directed road links. The data is speed only, which matters later for the control experiment. The single most important rule: a zero is an outage, not a traffic jam.");
}
{
  const s = slide("CONTENT", "01 The data");
  s.addText("Silence is not a traffic jam", { placeholder: "title" });
  fig(s, "eda/01_missingness", 0.6, 1.55, 8.3, 3.25);
  stat(s, 9.2, 1.55, 3.5, 1.55, eda.missingness.whole_network_outage_timestamps.toLocaleString("en-US"), "timestamps where ≥ 90% of sensors are silent", RED, 36);
  stat(s, 9.2, 3.25, 3.5, 1.55, `${pct(eda.splits.split_stats.test.missing_rate)}`, `missing in the test period vs ${pct(eda.splits.split_stats.train.missing_rate)} in training`, AMBER, 36);
  card(s, 0.6, 5.05, 12.1, 1.7);
  bullets(s, ["Inputs: forward-fill up to 30 minutes (past only), then the training-period average for that sensor and time of day, plus an is_missing flag",
    "Targets: never imputed — a model is only scored where the sensor actually reported"], 0.95, 5.3, 11.5, 1.4, 15);
  s.addNotes("The heatmap shows outages are clustered: whole-network blackouts on 61 days. The test period is noticeably noisier than training. Our rule: fill inputs with past-only information and flag them; never invent targets.");
}
{
  const s = slide("CONTENT", "01 The data");
  s.addText("The city breathes twice a day", { placeholder: "title" });
  fig(s, "eda/02_daily_profile", 0.6, 1.55, 7.6, 5.2);
  const dp = eda.daily_profile;
  stat(s, 8.5, 1.55, 4.2, 1.6, `${dp.weekday_min_mean_speed.toFixed(1)} mph`, "weekday low, around 17:30", RED, 38);
  stat(s, 8.5, 3.28, 4.2, 1.6, `${dp.weekend_min_mean_speed.toFixed(1)} mph`, "weekend low — rush hours vanish", TEAL, 38);
  stat(s, 8.5, 5.15, 4.2, 1.6, `${eda.heatmap.heatmap_min_mph.toFixed(1)}–${eda.heatmap.heatmap_max_mph.toFixed(1)}`, "mph range of weekly time-slot averages", AMBER, 34);
  s.addNotes("Two rush hours on weekdays, almost none at weekends. Across the week, time-slot averages range from about 42 to 66 mph. This is why every model gets time-of-day and day-of-week features.");
}
{
  const s = slide("CONTENT", "01 The data");
  s.addText("Turning roads into a graph", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.7, 5.2);
  T(s, "THRESHOLDED GAUSSIAN KERNEL", { x: 0.95, y: 1.85, w: 5.1, h: 0.35, fontSize: 14, bold: true, color: AMBER, charSpacing: 3 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.95, y: 2.35, w: 5.0, h: 0.95, fill: { color: "0A0F1E", transparency: 20 }, line: { color: SLATE, width: 0.75 }, rectRadius: 0.08 });
  T(s, "Wᵢⱼ = exp(−(dᵢⱼ / σ)²)   if ≥ κ, else 0", { x: 0.95, y: 2.35, w: 5.0, h: 0.95, fontSize: 20, fontFace: "Cambria", align: "center", valign: "middle" });
  bullets(s, ["dᵢⱼ = directed road distance (metres)", "σ = 2,584 m (spread of all distances), κ = 0.1",
    "Rebuilt matrix equals the published one to 1.0 × 10⁻⁷", "Phase 2 will use it in both directions, W + Wᵀ"], 0.95, 3.55, 5.1, 3.0, 16);
  fig(s, "eda/06_corr_vs_hops", 6.6, 1.55, 6.1, 3.45);
  const c = eda.corr_vs_hops;
  card(s, 6.6, 5.2, 6.1, 1.55);
  T(s, [{ text: `Neighbours really are related: `, options: { color: MUTED } },
        { text: `median correlation ${c.median_corr_1_hops.toFixed(2)} at 1 hop → ${c.median_corr_2_hops.toFixed(2)} at 2 hops → ${c.median_corr_8_hops.toFixed(2)} at 8 hops`, options: { bold: true } }],
    { x: 6.9, y: 5.4, w: 5.5, h: 1.2, fontSize: 16, valign: "middle" });
  s.addNotes("Distances become edge weights with a Gaussian kernel, exactly as in DCRNN, and our rebuild matches the published matrix to seven decimal places. On the right: sensor pairs one hop apart correlate much more strongly than distant ones. That is the case for a graph.");
}
{
  const s = slide("CONTENT", "01 The data");
  s.addText("Does congestion really travel upstream?", { placeholder: "title" });
  fig(s, "eda/07b_peak_lag_hist", 0.6, 1.55, 6.9, 3.95);
  const x = eda.xcorr;
  stat(s, 7.8, 1.55, 2.35, 1.9, pct(x.xcorr_peakhours_share_positive, 0), "of pairs: upstream reacts later", TEAL, 36);
  stat(s, 10.35, 1.55, 2.35, 1.9, pct(x.xcorr_peakhours_share_negative, 0), "of pairs: the opposite", RED, 36);
  card(s, 7.8, 3.65, 4.9, 1.85);
  T(s, `Peak hours, all ${x.xcorr_pairs.toLocaleString("en-US")} adjacent sensor pairs, lag measured in 5-minute steps.`, { x: 8.1, y: 3.85, w: 4.3, h: 1.5, fontSize: 15, color: MUTED });
  card(s, 0.6, 5.75, 12.1, 1.0, "3A1E28", 0);
  T(s, [{ text: "Honest reading: ", options: { bold: true, color: AMBER } }, { text: "spatial dependence is clear, directional propagation is only modestly supported. So the graph's benefit must be tested, not assumed." }],
    { x: 0.95, y: 5.75, w: 11.5, h: 1.0, fontSize: 16, valign: "middle" });
  s.addNotes(`We checked the hypothesis before building anything. For every adjacent pair we found the time lag of strongest correlation. At peak hours half the pairs show the upstream sensor reacting later, as the shockwave story predicts, but ${pct(x.xcorr_peakhours_share_negative, 0)} show the opposite. Real, but modest. That sets expectations for everything after.`);
}
// ================================================================= CHAPTER 2 PROTOCOL
chapter("02 A fair fight", "CHAPTER 02", "A fair fight", "No peeking into the future, the same test for every model, and statistics that respect time.",
  "Chapter two is about trust. Most impressive traffic results die here, through leakage.");
{
  const s = slide("CONTENT", "02 A fair fight");
  s.addText("No peeking into the future", { placeholder: "title" });
  const X = 0.6, Wt = 12.1, tot = 34272;
  const parts = [["TRAIN", 23990, "Mar 1 – May 23", "23,967 windows", TEAL], ["VAL", 3427, "May 23 – Jun 4", "3,404", AMBER], ["TEST", 6855, "Jun 4 – Jun 27", "6,832", RED]];
  let x = X;
  parts.forEach(([n, steps, dates, wins, col]) => {
    const w = (steps / tot) * Wt - 0.06;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.65, w, h: 0.85, fill: { color: col }, line: { color: col, transparency: 100 }, rectRadius: 0.08 });
    T(s, n, { x: x + 0.15, y: 1.65, w: Math.max(w - 0.3, 0.6), h: 0.85, fontSize: 16, bold: true, color: "0A0F1E", valign: "middle" });
    T(s, [{ text: dates.replace(' – ', ' –\n'), options: { breakLine: true, bold: true } }, { text: wins, options: { color: MUTED } }], { x, y: 2.6, w: Math.max(w, 1.0), h: 0.95, fontSize: 13 });
    x += w + 0.06;
  });
  const cardsY = 3.65, cw = 3.85;
  [["CHRONOLOGICAL", "Split by time 70/10/20. Windows are built inside each split, so no input or target crosses a boundary (unit-tested)."],
   ["TRAIN-ONLY STATISTICS", "Scaler 58.58 ± 12.82 mph, imputation means and the history table all come from training time only."],
   ["MASKED METRICS", "MAE, RMSE, MAPE in mph on observed targets; sliced by horizon, peak / off-peak / weekend, congested / free-flow."]]
    .forEach(([h, b], i) => {
      const cx = 0.6 + i * (cw + 0.275);
      card(s, cx, cardsY, cw, 3.1);
      T(s, h, { x: cx + 0.3, y: cardsY + 0.3, w: cw - 0.6, h: 0.4, fontSize: 14, bold: true, color: AMBER, charSpacing: 2 });
      T(s, b, { x: cx + 0.3, y: cardsY + 0.8, w: cw - 0.6, h: 2.1, fontSize: 16 });
    });
  s.addNotes("Our split is purely chronological. A random split would put near-identical windows in train and test and make every model look great. Everything learned from data — even the scaling constants — comes from training time only. A built-in alarm fires if any model gets below 3.0 mph MAE at 60 minutes, because that would almost certainly mean leakage. It never fired.");
}
{
  const s = slide("CONTENT", "02 A fair fight");
  s.addText("Statistics that respect time", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.9, 5.2);
  T(s, "DIEBOLD–MARIANO TEST", { x: 0.95, y: 1.85, w: 5.3, h: 0.35, fontSize: 14, bold: true, color: AMBER, charSpacing: 3 });
  bullets(s, ["Compares two models' error series hour by hour", "Errors are autocorrelated, so the variance uses a Newey–West (HAC) estimate",
    "Small-sample correction; for 3-seed models we report the worst p-value"], 0.95, 2.35, 5.3, 2.4, 16);
  T(s, "DAY-BLOCK BOOTSTRAP", { x: 0.95, y: 4.65, w: 5.3, h: 0.35, fontSize: 14, bold: true, color: TEAL, charSpacing: 3 });
  T(s, "Resample whole days (1,000×) for 95% intervals; paired for model differences.", { x: 0.95, y: 5.1, w: 5.3, h: 1.4, fontSize: 16 });
  // baselines chart
  const cats = ["15 min", "30 min", "60 min"];
  s.addChart(pres.charts.BAR, [
    { name: "Persistence", labels: cats, values: [3, 6, 12].map((h) => mt("persistence", h)) },
    { name: "Historical average", labels: cats, values: [3, 6, 12].map((h) => mt("historical_average", h)) },
    { name: "Seasonal naive", labels: cats, values: [3, 6, 12].map((h) => mt("seasonal_naive", h)) },
  ], chartBase({ x: 6.8, y: 1.55, w: 5.9, h: 4.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent2, HEX.accent5],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "b", valAxisMinVal: 0,
    showTitle: true, title: "Baselines to beat — test MAE (mph)" }));
  T(s, "Historical average 4.19 mph vs 4.16 in the DCRNN paper — an external check of our pipeline.", { x: 6.9, y: 6.0, w: 5.8, h: 0.75, fontSize: 14, color: MUTED });
  s.addNotes("Two statistical tools keep us honest. Diebold–Mariano tells us whether one model really beats another, with a variance that respects autocorrelation. The day-block bootstrap gives confidence intervals. On the right are the baselines. Persistence — just repeat the last speed — is surprisingly strong at 15 minutes. Our historical average lands within 0.03 of the published number, a nice sanity check.");
}
// ================================================================= CHAPTER 3 TREES
chapter("03 Trees", "CHAPTER 03", "Trees that read the clock", "Gradient-boosted trees with 45–59 hand-made features: how far does per-segment learning go?",
  "Chapter three: the strong classical baseline. If deep learning can't beat this, it hasn't earned its complexity.");
{
  const s = slide("CONTENT", "03 Trees");
  s.addText("Four tree models, one recipe", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.0, 5.2);
  T(s, "FEATURE RECIPE", { x: 0.95, y: 1.85, w: 4.4, h: 0.35, fontSize: 14, bold: true, color: AMBER, charSpacing: 3 });
  bullets(s, ["12 lags + rolling mean / std / min / max", "Missing flags, cyclic time-of-day and day-of-week", "Pooled: + sensor id, location, degree (50)",
    "Spatial: + upstream / downstream neighbour speeds (59)", "Optuna, 15 trials, validation only — absolute-error loss won"], 0.95, 2.35, 4.4, 4.2, 15);
  const cats = ["15 min", "30 min", "60 min"];
  s.addChart(pres.charts.BAR, ["xgb_per_segment", "xgb_global", "rf_multioutput", "xgb_spatial"].map((m, i) => ({
    name: ["XGB per-segment", "XGB pooled", "Random Forest", "XGB spatial"][i], labels: cats, values: [3, 6, 12].map((h) => mt(m, h)) })),
  chartBase({ x: 5.9, y: 1.55, w: 6.8, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent2, HEX.accent5, HEX.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, showLegend: true, legendPos: "b", valAxisMinVal: 0,
    showTitle: true, title: "Test MAE (mph) — lower is better" }));
  s.addNotes(`Four tree models. Per-segment XGBoost trains 621 small models, one per sensor and horizon. The pooled model learns across all sensors. The forest predicts all horizons at once. XGB-spatial adds hand-made neighbour features — our bridge between classical ML and graphs. XGB-spatial is best at 15 and 30 minutes (${f3(mt("xgb_spatial", 3))}, ${f3(mt("xgb_spatial", 6))}); per-segment wins at 60 minutes (${f3(mt("xgb_per_segment", 12))}). One caveat: every pooled model hit our 400-tree cap, so these numbers are conservative.`);
}
{
  const s = slide("CONTENT", "03 Trees");
  s.addText("The Phase 1 scoreboard", { placeholder: "title" });
  const H = Array.from({ length: 12 }, (_, i) => `${(i + 1) * 5}`);
  const ser = [["persistence", "Persistence"], ["historical_average", "Historical average"], ["rf_multioutput", "Random Forest"], ["xgb_per_segment", "XGB per-segment"], ["xgb_global", "XGB pooled"], ["xgb_spatial", "XGB + neighbours"]];
  const curves = ser.map(([k, n]) => ({ name: n, labels: H, values: D.curves[k].map((v) => (v == null ? null : v)) }));
  s.addChart(pres.charts.LINE, curves, chartBase({ x: 0.6, y: 1.55, w: 7.9, h: 5.2, chartColors: [HEX.accent5, HEX.accent2, "9AA7C7", HEX.accent4, "FF8A3D", HEX.accent3],
    lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 5, showLegend: true, legendPos: "b", valAxisMinVal: 2, valAxisMaxVal: 6, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0.0",
    showTitle: true, title: "Test MAE (mph) by horizon (minutes)" }));
  [["15 MIN", "xgb_spatial", 3], ["30 MIN", "xgb_spatial", 6], ["60 MIN", "xgb_per_segment", 12]].forEach(([h, m, hz], i) => {
    const y = 1.55 + i * 1.78;
    card(s, 8.8, y, 3.9, 1.58);
    T(s, h, { x: 9.1, y: y + 0.2, w: 1.4, h: 0.4, fontSize: 14, bold: true, color: AMBER, charSpacing: 2 });
    T(s, `${f2(mt(m, hz))} mph`, { x: 9.1, y: y + 0.55, w: 3.4, h: 0.7, fontSize: 34, bold: true, fontFace: "Arial" });
    T(s, `best tree · persistence ${f2(mt("persistence", hz))}`, { x: 9.1, y: y + 1.15, w: 3.4, h: 0.35, fontSize: 13, color: MUTED });
  });
  s.addNotes(`Every learned model beats persistence. XGBoost with neighbour features is best at 15 and 30 minutes (${f3(mt("xgb_spatial", 3))}, ${f3(mt("xgb_spatial", 6))}); per-segment XGBoost wins at 60 minutes (${f3(mt("xgb_per_segment", 12))}). The Random Forest is the weakest tree model, and at an hour ahead it loses even to the historical average. No model gets near the 3.0 mph leakage alarm.`);
}
{
  const s = slide("CONTENT", "03 Trees");
  s.addText("Do neighbours help the trees?", { placeholder: "title" });
  const sl = ["all", "peak", "offpeak_weekday", "congested"], lab = ["All", "Peak", "Off-peak", "Congested"];
  s.addChart(pres.charts.BAR, [3, 6, 12].map((h) => ({ name: `${h * 5} min`, labels: lab, values: sl.map((x) => D.gap[`${x}|${h}`]) })),
    chartBase({ x: 0.6, y: 1.55, w: 7.6, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent3, HEX.accent2, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, showLegend: true, legendPos: "b",
      showTitle: true, title: "MAE gain from neighbour features (mph, pooled − spatial)" }));
  stat(s, 8.5, 1.55, 4.2, 2.0, `+${f2(D.gap["congested|6"])} mph`, "gain in congestion at 30 min — 3–4× the overall gain", RED, 40);
  card(s, 8.5, 3.75, 4.2, 3.0);
  bullets(s, ["Significant in almost every slice (DM test, paired bootstrap)", "Biggest at peaks and in jams — as the hypothesis predicts",
    { t: "But the gain does not grow with horizon", o: { bold: true, color: AMBER } }], 8.8, 4.0, 3.7, 2.6, 15);
  s.addNotes("The bridge experiment. Adding neighbour features to the same XGBoost improves accuracy almost everywhere, and the improvement is concentrated where we expected: peaks and congestion. But the hypothesis also said the gain should grow with horizon — it doesn't, it peaks at 30 minutes. Half a win, reported honestly.");
}
{
  const s = slide("CONTENT", "03 Trees");
  s.addText("What the trees actually look at", { placeholder: "title" });
  fig(s, "ml/shap_xgb_spatial_h12", 0.6, 1.55, 5.55, 5.2);
  card(s, 6.45, 1.55, 6.25, 5.2);
  T(s, "TREESHAP, 5,000 TEST ROWS", { x: 6.8, y: 1.85, w: 5.6, h: 0.35, fontSize: 14, bold: true, color: AMBER, charSpacing: 3 });
  bullets(s, ["15 min: the last half-hour average and the latest speed dominate", "60 min: sensor identity and time of day take over",
    "Neighbour speeds appear in the top features, but never dominate", "Same-time-yesterday lag adds nothing (2.879 vs 2.880 mph)"], 6.8, 2.35, 5.6, 4.2, 16);
  s.addNotes("SHAP shows how the trees think. Short horizon: what is happening right now. Long horizon: which sensor this is and what time it is — essentially a learned timetable. Neighbour information is used, but as a supporting feature. And a 'same time yesterday' feature added nothing measurable.");
}
// ================================================================= CLOSING
{
  const s = slide("CONTENT", "Closing");
  s.addText("What Phase 1 taught us", { placeholder: "title" });
  const items = [
    ["01", "Leak-free or it doesn't count", "Chronological splits, train-only statistics and masked metrics — the leakage alarm never fired.", AMBER],
    ["02", "Trees are a strong baseline", `Every tree beats persistence; best ${f2(mt("xgb_spatial", 3))} · ${f2(mt("xgb_spatial", 6))} · ${f2(mt("xgb_per_segment", 12))} mph at 15/30/60 min.`, TEAL],
    ["03", "Neighbours help where it hurts", "Neighbour features pay off at peaks and in jams — not more at longer horizons.", RED],
    ["04", "Time and identity dominate", "At an hour ahead the trees mostly use the sensor and the clock; neighbours are a supporting actor.", INDIGO],
    ["05", "Evidence for the graph is modest", "Spatial dependence is clear; upstream propagation is only weakly asymmetric (50% vs 38%).", SLATE],
  ];
  items.forEach(([n, h, b, col], i) => {
    const y = 1.5 + i * 1.07;
    card(s, 0.6, y, 12.1, 0.93);
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: y + 0.14, w: 0.65, h: 0.65, fill: { color: col }, line: { color: col, transparency: 100 } });
    T(s, n, { x: 0.85, y: y + 0.14, w: 0.65, h: 0.65, fontSize: 15, bold: true, color: "0A0F1E", align: "center", valign: "middle" });
    T(s, h, { x: 1.8, y: y + 0.1, w: 4.4, h: 0.73, fontSize: 18, bold: true, valign: "middle" });
    T(s, b, { x: 6.3, y: y + 0.1, w: 6.2, h: 0.73, fontSize: 15, color: MUTED, valign: "middle" });
  });
  s.addNotes("Five takeaways from Phase 1. Trust first. Trees are a strong baseline. Neighbours help in jams. At long horizons the trees mostly use identity and time. And the evidence that a learned graph will help is real but modest, which is exactly what Phase 2 tests.");
}
{
  const s = slide("CONTENT", "Closing");
  s.addText("Limits, and what Phase 2 tests", { placeholder: "title" });
  card(s, 0.6, 1.55, 5.9, 5.2);
  T(s, "LIMITATIONS", { x: 0.95, y: 1.85, w: 5.3, h: 0.35, fontSize: 14, bold: true, color: RED, charSpacing: 3 });
  bullets(s, ["One city, one season (Mar–Jun 2012), speed only", "Laptop budgets: 15 tuning trials, pooled models on 25% of rows, 400-tree cap always reached",
    "Hand-made neighbour features, fixed distance graph", "Test period has more sensor outages than training"], 0.95, 2.45, 5.3, 4.2, 20);
  card(s, 6.8, 1.55, 5.9, 5.2);
  T(s, "PHASE 2", { x: 7.15, y: 1.85, w: 5.3, h: 0.35, fontSize: 14, bold: true, color: TEAL, charSpacing: 3 });
  bullets(s, ["Learn the neighbour weighting: graph attention (STGAT)", "A no-graph twin to isolate the graph's effect",
    "A leakage-safe hybrid of trees and graph", "Ablations, maps and a ramp-metering simulation"], 7.15, 2.45, 5.3, 4.2, 20);
  s.addNotes("Be clear about limits: most are compute budget, not method. Then hand over to Phase 2: replace hand-made neighbour features with learned attention, test it against a no-graph twin, and combine it with the trees.");
}
{
  const s = slide("TITLE", "Closing");
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions welcome — every number is in the repository.", { placeholder: "body" });
  T(s, [{ text: "TEAM T042", options: { bold: true, color: AMBER, charSpacing: 3, breakLine: true } },
        { text: "Pranav Singh · Kunal · Himanshu · Kanchan · Vaibhav", options: { breakLine: true } },
        { text: "github.com/Pranav-Singh-Devloper/traffic-oracle", options: { color: MUTED, breakLine: true } },
        { text: "Live demo: traffic-oracle.streamlit.app", options: { color: TEAL, bold: true } }],
    { x: 0.8, y: 3.3, w: 11, h: 1.4, fontSize: 18 });
  s.addNotes("Close Phase 1 and invite questions.");
}

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
