import test from 'node:test';
import assert from 'node:assert/strict';
import { SCENARIOS, makeFixture, stabilityMetric, reviewFixture, exportFixture, fixtureCSV } from './model.mjs';
import { readFileSync } from 'node:fs';

const close = (a, b, tolerance = 1e-12) => assert.ok(Math.abs(a - b) <= tolerance, `${a} ≠ ${b}`);

test('public metric has the exact mean, decline and population residual penalties', () => {
  const m = stabilityMetric([0.7, 0.7, 0.6, 0.6]);
  close(m.mean, 0.65);
  close(m.slope, -0.04);
  close(m.residualStd, Math.sqrt(0.0005));
  close(m.stability, 0.65 - 3.52 - 0.5 * Math.sqrt(0.0005));
});
test('positive trends receive no bonus and flat fixtures have no penalty', () => {
  close(stabilityMetric([0.3, 0.4, 0.5]).stability, 0.4);
  close(stabilityMetric([0.6, 0.6, 0.6]).stability, 0.6);
});
test('metric rejects empty, singleton, missing, nonfinite and out-of-domain data', () => {
  for (const invalid of [[], [0.5], [0.5, NaN], [0.5, Infinity], [0.5, 2], null, 'data']) {
    assert.throws(() => stabilityMetric(invalid), TypeError);
  }
});
test('presets are deterministic with intended mean, slope and residual variation', () => {
  for (const preset of Object.values(SCENARIOS)) {
    const rows = makeFixture(preset);
    assert.deepEqual(rows, makeFixture(preset));
    assert.equal(rows.length, 24);
    assert.deepEqual(rows.map(row => row.week), Array.from({ length: 24 }, (_, i) => i + 1));
    const metric = stabilityMetric(rows.map(row => row.gini));
    close(metric.mean, 0.6);
    close(metric.slope, -preset.decline);
    close(metric.residualStd, preset.variation);
  }
});
test('all slider extremes remain visible in the fixed chart domain', () => {
  for (const decline of [0, 0.003]) for (const variation of [0, 0.09]) {
    const rows = makeFixture({ decline, variation });
    assert.ok(rows.every(row => row.gini >= 0.35 && row.gini <= 0.85));
    const metric = stabilityMetric(rows.map(row => row.gini));
    close(metric.mean, 0.6);
    close(metric.slope, -decline);
    close(metric.residualStd, variation);
  }
});
test('fixture bounds reject invalid settings instead of silently clipping them', () => {
  for (const settings of [{ decline: -1 }, { decline: 0.004 }, { variation: -1 }, { variation: 0.1 }, { variation: NaN }]) {
    assert.throws(() => makeFixture(settings), RangeError);
  }
});
test('illustrative review distinguishes all three scenarios without leaking real gates', () => {
  const result = Object.fromEntries(Object.entries(SCENARIOS).map(([key, preset]) => [key, reviewFixture(stabilityMetric(makeFixture(preset).map(row => row.gini)))]));
  assert.equal(result.stable.passed, true);
  assert.equal(result.drift.passed, false);
  assert.equal(result.volatile.passed, false);
  assert.equal(result.drift.checks.find(c => c.key === 'decline').passed, false);
  assert.equal(result.volatile.checks.find(c => c.key === 'variation').passed, false);
});
test('CSV and JSON exports preserve the chosen fixture and mark it synthetic', () => {
  const settings = { decline: 0.0015, variation: 0.025 };
  const json = JSON.parse(JSON.stringify(exportFixture(settings)));
  assert.equal(json.weekly.length, 24);
  assert.match(json.dataType, /synthetic/);
  assert.deepEqual(json.parameters, settings);
  close(json.metric.stability, stabilityMetric(json.weekly.map(row => row.gini)).stability);
  const csv = fixtureCSV(settings).trim().split('\n');
  assert.equal(csv.length, 25);
  assert.equal(csv[0], 'data_type,week,synthetic_weekly_gini,fitted_trend');
  for (let i = 1; i < csv.length; i += 1) {
    const row = csv[i].split(',');
    assert.equal(row[0], 'synthetic');
    assert.equal(Number(row[1]), i);
    close(Number(row[2]), json.weekly[i - 1].gini, 1e-9);
  }
});
test('document controls, data table and accessible SVG targets are present', () => {
  const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length, 'IDs must be unique');
  const script = readFileSync(new URL('./app.mjs', import.meta.url), 'utf8');
  for (const [, id] of script.matchAll(/\$\('([^']+)'\)/g)) assert.ok(ids.includes(id), `missing #${id}`);
  assert.match(html, /for="decline"/);
  assert.match(html, /for="variation"/);
  assert.match(html, /role="status"/);
  assert.match(html, /aria-labelledby="chart-title chart-desc"/);
  assert.match(html, /<noscript>/);
  assert.match(html, /Synthetic demonstration/);
  assert.doesNotMatch(html, /<(?:iframe|form)\b/i);
  assert.doesNotMatch(script, /\b(?:fetch|XMLHttpRequest|localStorage)\b/);
});
