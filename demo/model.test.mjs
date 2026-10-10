import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SCENARIOS,
  makeFixture,
  stabilityMetric,
  reviewFixture,
  exportFixture,
  fixtureCSV,
  compareFixtures,
  comparisonCSV,
} from './model.mjs';
import { readFileSync } from 'node:fs';

const close = (a, b, tolerance = 1e-12) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} ≠ ${b}`);

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
  for (const invalid of [
    [],
    [0.5],
    [0.5, NaN],
    [0.5, Infinity],
    [0.5, 2],
    [0.5, true],
    new Array(3),
    [0.5, , 0.6],
    null,
    'data',
  ]) {
    assert.throws(() => stabilityMetric(invalid), TypeError);
  }
});
test('presets are deterministic with intended mean, slope and residual variation', () => {
  for (const preset of Object.values(SCENARIOS)) {
    const rows = makeFixture(preset);
    assert.deepEqual(rows, makeFixture(preset));
    assert.equal(rows.length, 24);
    assert.deepEqual(
      rows.map((row) => row.week),
      Array.from({ length: 24 }, (_, i) => i + 1),
    );
    const metric = stabilityMetric(rows.map((row) => row.gini));
    close(metric.mean, 0.6);
    close(metric.slope, -preset.decline);
    close(metric.residualStd, preset.variation);
  }
});
test('all slider extremes remain visible in the fixed chart domain', () => {
  for (const decline of [0, 0.003])
    for (const variation of [0, 0.09]) {
      const rows = makeFixture({ decline, variation });
      assert.ok(rows.every((row) => row.gini >= 0.35 && row.gini <= 0.85));
      const metric = stabilityMetric(rows.map((row) => row.gini));
      close(metric.mean, 0.6);
      close(metric.slope, -decline);
      close(metric.residualStd, variation);
    }
});
test('fixture bounds reject invalid settings instead of silently clipping them', () => {
  for (const settings of [
    { decline: -1 },
    { decline: 0.004 },
    { variation: -1 },
    { variation: 0.1 },
    { variation: NaN },
  ]) {
    assert.throws(() => makeFixture(settings), RangeError);
  }
});
test('illustrative review distinguishes all three scenarios without leaking real gates', () => {
  const result = Object.fromEntries(
    Object.entries(SCENARIOS).map(([key, preset]) => [
      key,
      reviewFixture(
        stabilityMetric(makeFixture(preset).map((row) => row.gini)),
      ),
    ]),
  );
  assert.equal(result.stable.passed, true);
  assert.equal(result.drift.passed, false);
  assert.equal(result.volatile.passed, false);
  assert.equal(
    result.drift.checks.find((c) => c.key === 'decline').passed,
    false,
  );
  assert.equal(
    result.volatile.checks.find((c) => c.key === 'variation').passed,
    false,
  );
});
test('CSV and JSON exports preserve the chosen fixture and mark it synthetic', () => {
  const settings = { decline: 0.0015, variation: 0.025 };
  const json = JSON.parse(JSON.stringify(exportFixture(settings)));
  assert.equal(json.weekly.length, 24);
  assert.match(json.dataType, /synthetic/);
  assert.deepEqual(json.parameters, settings);
  close(
    json.metric.stability,
    stabilityMetric(json.weekly.map((row) => row.gini)).stability,
  );
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
test('default and partial-setting exports describe the exact numeric parameters actually used', () => {
  for (const [settings, expected] of [
    [undefined, { decline: 0.00025, variation: 0.014 }],
    [{}, { decline: 0.00025, variation: 0.014 }],
    [{ variation: 0.02 }, { decline: 0.00025, variation: 0.02 }],
    [{ decline: 0 }, { decline: 0, variation: 0.014 }],
    [SCENARIOS.drift, { decline: 0.002, variation: 0.014 }],
  ]) {
    const result = exportFixture(settings);
    assert.deepEqual(result.parameters, expected);
    assert.equal(result.evidence_type, 'SYNTHETIC_ONLY');
    close(result.metric.slope, -expected.decline);
    close(result.metric.residualStd, expected.variation);
    assert.deepEqual(result.weekly, makeFixture(expected));
    assert.equal(fixtureCSV(settings).trim().split('\n').length, 25);
  }
});
test('unknown settings and malformed names cannot silently fall back to another scenario', () => {
  for (const settings of [
    null,
    true,
    1,
    [],
    new Date(0),
    { declnie: 0.001 },
    { variation: 0.02, privateParameter: 1 },
    { name: '' },
    { name: 3 },
    { name: 'a'.repeat(81) },
  ]) {
    for (const fn of [makeFixture, exportFixture, fixtureCSV])
      assert.throws(() => fn(settings), TypeError);
    assert.throws(() => compareFixtures(settings, SCENARIOS.stable), TypeError);
    assert.throws(() => compareFixtures(SCENARIOS.stable, settings), TypeError);
  }
  for (const settings of [
    { decline: '0.001' },
    { variation: true },
    { decline: Infinity },
    { variation: -0.01 },
  ]) {
    assert.throws(
      () => compareFixtures(SCENARIOS.stable, settings),
      RangeError,
    );
  }
  assert.deepEqual(
    makeFixture({ ...SCENARIOS.stable, name: 'Named public scenario' }),
    makeFixture(SCENARIOS.stable),
  );
});
test('illustrative review rejects missing, nonfinite and negative variation metrics', () => {
  for (const metric of [
    null,
    {},
    { stability: NaN, slope: 0, residualStd: 0 },
    { stability: 0.6, slope: '0', residualStd: 0 },
    { stability: 0.6, slope: 0, residualStd: -0.01 },
  ]) {
    assert.throws(() => reviewFixture(metric), TypeError);
  }
});
test('paired identical settings preserve every week and give an exact zero comparison', () => {
  const result = compareFixtures(SCENARIOS.stable, SCENARIOS.stable);
  assert.equal(result.evidence_type, 'SYNTHETIC_ONLY');
  assert.equal(result.direction, 'unchanged');
  assert.equal(result.weekly.length, 24);
  assert.deepEqual(result.delta, {
    mean: 0,
    slope: 0,
    intercept: 0,
    residualStd: 0,
    trendPenalty: 0,
    variationPenalty: 0,
    stability: 0,
  });
  assert.ok(
    result.weekly.every(
      (row) =>
        row.baselineGini === row.candidateGini &&
        row.deltaGini === 0 &&
        row.deltaTrend === 0,
    ),
  );
  assert.deepEqual(result.checks, {
    sameWeeks: true,
    weeklyDeltaZeroMean: true,
    scoreDeltaEqualsComponents: true,
  });
});
test('pure decline comparison matches the independent penalty and centered-week hand formula', () => {
  const a = { decline: 0.00025, variation: 0.014 };
  const b = { decline: 0.002, variation: 0.014 };
  const result = compareFixtures(a, b);
  const difference = b.decline - a.decline;
  close(result.delta.mean, 0);
  close(result.delta.slope, -difference);
  close(result.delta.trendPenalty, -88 * difference);
  close(result.delta.variationPenalty, 0);
  close(result.delta.stability, -88 * difference);
  assert.equal(result.direction, 'lower');
  for (const row of result.weekly) {
    close(row.deltaGini, -difference * (row.week - 12.5));
    close(row.deltaTrend, row.deltaGini);
  }
  const reverse = compareFixtures(b, a);
  assert.equal(reverse.direction, 'higher');
  for (const key of Object.keys(result.delta))
    close(reverse.delta[key], -result.delta[key]);
});
test('pure variation comparison conserves the mean and trend while changing only the residual penalty', () => {
  const a = { decline: 0.00025, variation: 0.01 };
  const b = { decline: 0.00025, variation: 0.06 };
  const result = compareFixtures(a, b);
  close(result.delta.mean, 0);
  close(result.delta.slope, 0);
  close(result.delta.trendPenalty, 0);
  close(result.delta.residualStd, 0.05);
  close(result.delta.variationPenalty, -0.025);
  close(result.delta.stability, -0.025);
  const weeklyDifference = stabilityMetric(
    result.weekly.map((row) => row.deltaGini),
  );
  close(weeklyDifference.mean, 0);
  close(weeklyDifference.slope, 0);
  close(weeklyDifference.residualStd, 0.05);
});
test('mixed comparison conserves weekly alignment and reconciles all additive score components', () => {
  for (const baseline of Object.values(SCENARIOS))
    for (const candidate of Object.values(SCENARIOS)) {
      const result = compareFixtures(baseline, candidate);
      assert.deepEqual(
        result.weekly.map((row) => row.week),
        Array.from({ length: 24 }, (_, i) => i + 1),
      );
      close(
        result.weekly.reduce((sum, row) => sum + row.deltaGini, 0),
        0,
      );
      close(
        result.delta.stability,
        88 * (baseline.decline - candidate.decline) +
          0.5 * (baseline.variation - candidate.variation),
      );
      close(
        result.components.reduce((sum, row) => sum + row.delta, 0),
        result.delta.stability,
      );
      assert.deepEqual(
        result.components.map((row) => row.key),
        ['mean', 'trendPenalty', 'variationPenalty'],
      );
    }
});
test('comparison CSV and JSON export the same pinned and candidate values with declared rounding', () => {
  const baseline = SCENARIOS.stable,
    candidate = { decline: 0.0018, variation: 0.035 };
  const result = JSON.parse(
    JSON.stringify(compareFixtures(baseline, candidate)),
  );
  const csv = comparisonCSV(baseline, candidate).trim().split('\n');
  assert.equal(csv.length, 25);
  assert.equal(
    csv[0],
    'data_type,week,baseline_synthetic_weekly_gini,candidate_synthetic_weekly_gini,delta_gini,baseline_fitted_trend,candidate_fitted_trend,delta_fitted_trend',
  );
  const fields = [
    'baselineGini',
    'candidateGini',
    'deltaGini',
    'baselineTrend',
    'candidateTrend',
    'deltaTrend',
  ];
  csv.slice(1).forEach((line, index) => {
    const values = line.split(',');
    assert.equal(values[0], 'synthetic_comparison');
    assert.equal(Number(values[1]), index + 1);
    fields.forEach((key, column) =>
      close(Number(values[column + 2]), result.weekly[index][key], 5.1e-11),
    );
    close(Number(values[3]) - Number(values[2]), Number(values[4]), 1.1e-10);
  });
  close(
    result.candidate.metric.stability - result.baseline.metric.stability,
    result.delta.stability,
  );
  assert.match(result.comparisonScope, /not evidence of a better credit model/);
});
test('comparison snapshots parameters without mutating or retaining editable input references', () => {
  const baseline = { ...SCENARIOS.stable },
    candidate = { ...SCENARIOS.drift };
  const before = JSON.stringify([baseline, candidate]);
  const result = compareFixtures(baseline, candidate),
    snapshot = JSON.stringify(result);
  assert.equal(JSON.stringify([baseline, candidate]), before);
  assert.equal(JSON.stringify(compareFixtures(baseline, candidate)), snapshot);
  baseline.decline = 0.003;
  candidate.variation = 0.09;
  assert.equal(JSON.stringify(result), snapshot);
  assert.notEqual(result.baseline.weekly, result.candidate.weekly);
});
test('document controls, data table and accessible SVG targets are present', () => {
  const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length, 'IDs must be unique');
  const script = readFileSync(new URL('./app.mjs', import.meta.url), 'utf8');
  for (const [, id] of script.matchAll(/\$\('([^']+)'\)/g))
    assert.ok(ids.includes(id), `missing #${id}`);
  assert.match(html, /for="decline"/);
  assert.match(html, /for="variation"/);
  assert.match(html, /role="status"/);
  assert.match(html, /aria-labelledby="chart-title chart-desc"/);
  assert.match(html, /<noscript\s*>/);
  assert.match(html, /Synthetic demonstration/);
  assert.doesNotMatch(html, /<(?:iframe|form)\b/i);
  assert.doesNotMatch(script, /\b(?:fetch|XMLHttpRequest|localStorage)\b/);
});
