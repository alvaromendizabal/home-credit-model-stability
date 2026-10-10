/** Public, deterministic illustration. No applicant data or trained model. */
export const SCENARIOS = Object.freeze({
  stable: Object.freeze({ name: 'Steady performance', decline: 0.00025, variation: 0.014 }),
  drift: Object.freeze({ name: 'Gradual decline', decline: 0.002, variation: 0.014 }),
  volatile: Object.freeze({ name: 'Uneven performance', decline: 0.00025, variation: 0.065 }),
});
export const ILLUSTRATIVE_THRESHOLDS = Object.freeze({ stability: 0.55, decline: 0.001, variation: 0.04 });

/** OLS uses ordinal, equally spaced week positions; residual std is population std. */
export function stabilityMetric(ginis) {
  if (!Array.isArray(ginis) || ginis.length < 2 || ginis.some(x => !Number.isFinite(x) || x < -1 || x > 1)) {
    throw new TypeError('Provide at least two finite weekly Gini values in [-1, 1].');
  }
  const n = ginis.length;
  const mean = ginis.reduce((a, b) => a + b, 0) / n;
  const center = (n - 1) / 2;
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < n; i += 1) {
    numerator += (i - center) * (ginis[i] - mean);
    denominator += (i - center) ** 2;
  }
  const slope = numerator / denominator;
  const intercept = mean - slope * center;
  const residualStd = Math.sqrt(ginis.reduce((sum, g, i) => sum + (g - intercept - slope * i) ** 2, 0) / n);
  const trendPenalty = 88 * Math.min(0, slope);
  const variationPenalty = -0.5 * residualStd;
  return { mean, slope, intercept, residualStd, trendPenalty, variationPenalty, stability: mean + trendPenalty + variationPenalty };
}

/** Noise has zero mean, zero OLS slope, and unit population residual std. */
export function makeFixture({ decline = 0.00025, variation = 0.014 } = {}) {
  if (!Number.isFinite(decline) || decline < 0 || decline > 0.003 || !Number.isFinite(variation) || variation < 0 || variation > 0.09) {
    throw new RangeError('Decline must be 0–0.003 and variation must be 0–0.09.');
  }
  const count = 24;
  const raw = Array.from({ length: count }, (_, i) => 0.55 * Math.sin(i * 2.31 + 0.5) + 0.28 * Math.cos(i * 0.87) + 0.1 * Math.sin(i * 4.12));
  const stats = stabilityMetric(raw);
  return raw.map((v, i) => ({
    week: i + 1,
    gini: 0.6 - decline * (i - (count - 1) / 2) + variation * (v - stats.intercept - stats.slope * i) / stats.residualStd,
  }));
}

export function reviewFixture(metric) {
  const tolerance = 1e-12;
  const checks = [
    { key: 'stability', label: 'Stability score', passed: metric.stability + tolerance >= ILLUSTRATIVE_THRESHOLDS.stability, threshold: 'At least 0.550' },
    { key: 'decline', label: 'Weekly decline', passed: -metric.slope <= ILLUSTRATIVE_THRESHOLDS.decline + tolerance, threshold: 'At most 0.0010' },
    { key: 'variation', label: 'Residual variation', passed: metric.residualStd <= ILLUSTRATIVE_THRESHOLDS.variation + tolerance, threshold: 'At most 0.040' },
  ];
  return { passed: checks.every(check => check.passed), checks };
}

export function exportFixture(settings) {
  const weekly = makeFixture(settings);
  const metric = stabilityMetric(weekly.map(row => row.gini));
  return {
    schema: 'home-credit-stability-illustration-v1',
    dataType: 'Deterministic synthetic weekly Gini fixture; not model inference or applicant data.',
    formula: 'mean(weekly_gini) + 88 * min(0, OLS_slope) - 0.5 * population_std(OLS_residuals)',
    parameters: { decline: settings.decline, variation: settings.variation },
    metric,
    illustrativeReview: reviewFixture(metric),
    weekly,
  };
}

export function fixtureCSV(settings) {
  const fixture = exportFixture(settings);
  return 'data_type,week,synthetic_weekly_gini,fitted_trend\n' + fixture.weekly.map((row, i) => `synthetic,${row.week},${row.gini.toFixed(10)},${(fixture.metric.intercept + fixture.metric.slope * i).toFixed(10)}`).join('\n') + '\n';
}
