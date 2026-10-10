/** Public, deterministic illustration. No applicant data or trained model. */
export const SCENARIOS = Object.freeze({
  stable: Object.freeze({
    name: 'Steady performance',
    decline: 0.00025,
    variation: 0.014,
  }),
  drift: Object.freeze({
    name: 'Gradual decline',
    decline: 0.002,
    variation: 0.014,
  }),
  volatile: Object.freeze({
    name: 'Uneven performance',
    decline: 0.00025,
    variation: 0.065,
  }),
});
export const ILLUSTRATIVE_THRESHOLDS = Object.freeze({
  stability: 0.55,
  decline: 0.001,
  variation: 0.04,
});

/** Named presets remain accepted; every numeric parameter is explicit in exports. */
function fixtureSettings(settings = {}) {
  if (
    settings === null ||
    typeof settings !== 'object' ||
    Array.isArray(settings) ||
    Object.prototype.toString.call(settings) !== '[object Object]'
  ) {
    throw new TypeError('Fixture settings must be an object.');
  }
  if (
    Object.keys(settings).some(
      (key) => !['decline', 'variation', 'name'].includes(key),
    )
  ) {
    throw new TypeError(
      'Unknown fixture setting; use decline, variation, and optional name.',
    );
  }
  if (
    'name' in settings &&
    (typeof settings.name !== 'string' ||
      !settings.name.trim() ||
      settings.name.length > 80)
  ) {
    throw new TypeError(
      'The optional scenario name must be a nonempty string of at most 80 characters.',
    );
  }
  const decline =
    settings.decline === undefined
      ? SCENARIOS.stable.decline
      : settings.decline;
  const variation =
    settings.variation === undefined
      ? SCENARIOS.stable.variation
      : settings.variation;
  if (
    !Number.isFinite(decline) ||
    decline < 0 ||
    decline > 0.003 ||
    !Number.isFinite(variation) ||
    variation < 0 ||
    variation > 0.09
  ) {
    throw new RangeError(
      'Decline must be 0–0.003 and variation must be 0–0.09.',
    );
  }
  return { decline, variation };
}

/** OLS uses ordinal, equally spaced week positions; residual std is population std. */
export function stabilityMetric(ginis) {
  if (!Array.isArray(ginis) || ginis.length < 2) {
    throw new TypeError(
      'Provide at least two finite weekly Gini values in [-1, 1].',
    );
  }
  // Array.some skips missing slots; iterate every position to reject sparse input.
  for (let i = 0; i < ginis.length; i += 1) {
    if (
      !Object.prototype.hasOwnProperty.call(ginis, i) ||
      !Number.isFinite(ginis[i]) ||
      ginis[i] < -1 ||
      ginis[i] > 1
    ) {
      throw new TypeError(
        'Provide at least two finite weekly Gini values in [-1, 1].',
      );
    }
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
  const residualStd = Math.sqrt(
    ginis.reduce((sum, g, i) => sum + (g - intercept - slope * i) ** 2, 0) / n,
  );
  const trendPenalty = 88 * Math.min(0, slope);
  const variationPenalty = -0.5 * residualStd;
  return {
    mean,
    slope,
    intercept,
    residualStd,
    trendPenalty,
    variationPenalty,
    stability: mean + trendPenalty + variationPenalty,
  };
}

/** Noise has zero mean, zero OLS slope, and unit population residual std. */
export function makeFixture(settings = {}) {
  const { decline, variation } = fixtureSettings(settings);
  const count = 24;
  const raw = Array.from(
    { length: count },
    (_, i) =>
      0.55 * Math.sin(i * 2.31 + 0.5) +
      0.28 * Math.cos(i * 0.87) +
      0.1 * Math.sin(i * 4.12),
  );
  const stats = stabilityMetric(raw);
  return raw.map((v, i) => ({
    week: i + 1,
    gini:
      0.6 -
      decline * (i - (count - 1) / 2) +
      (variation * (v - stats.intercept - stats.slope * i)) / stats.residualStd,
  }));
}

export function reviewFixture(metric) {
  if (
    !metric ||
    typeof metric !== 'object' ||
    !['stability', 'slope', 'residualStd'].every((key) =>
      Number.isFinite(metric[key]),
    ) ||
    metric.residualStd < 0
  ) {
    throw new TypeError(
      'Review requires finite stability, slope and nonnegative residual variation.',
    );
  }
  const tolerance = 1e-12;
  const checks = [
    {
      key: 'stability',
      label: 'Stability score',
      passed: metric.stability + tolerance >= ILLUSTRATIVE_THRESHOLDS.stability,
      threshold: 'At least 0.550',
    },
    {
      key: 'decline',
      label: 'Weekly decline',
      passed: -metric.slope <= ILLUSTRATIVE_THRESHOLDS.decline + tolerance,
      threshold: 'At most 0.0010',
    },
    {
      key: 'variation',
      label: 'Residual variation',
      passed:
        metric.residualStd <= ILLUSTRATIVE_THRESHOLDS.variation + tolerance,
      threshold: 'At most 0.040',
    },
  ];
  return { passed: checks.every((check) => check.passed), checks };
}

export function exportFixture(settings = {}) {
  const parameters = fixtureSettings(settings);
  const weekly = makeFixture(parameters);
  const metric = stabilityMetric(weekly.map((row) => row.gini));
  return {
    schema: 'home-credit-stability-illustration-v1',
    evidence_type: 'SYNTHETIC_ONLY',
    dataType:
      'Deterministic synthetic weekly Gini fixture; not model inference or applicant data.',
    formula:
      'mean(weekly_gini) + 88 * min(0, OLS_slope) - 0.5 * population_std(OLS_residuals)',
    parameters,
    metric,
    illustrativeReview: reviewFixture(metric),
    weekly,
  };
}

export function fixtureCSV(settings) {
  const fixture = exportFixture(settings);
  return (
    'data_type,week,synthetic_weekly_gini,fitted_trend\n' +
    fixture.weekly
      .map(
        (row, i) =>
          `synthetic,${row.week},${row.gini.toFixed(10)},${(fixture.metric.intercept + fixture.metric.slope * i).toFixed(10)}`,
      )
      .join('\n') +
    '\n'
  );
}

/** Paired synthetic scenarios share the same 24 weeks and residual pattern. */
export function compareFixtures(settingsA = {}, settingsB = {}) {
  const baseline = exportFixture(settingsA);
  const candidate = exportFixture(settingsB);
  const delta = Object.fromEntries(
    Object.keys(baseline.metric).map((key) => [
      key,
      candidate.metric[key] - baseline.metric[key],
    ]),
  );
  const labels = {
    mean: 'Mean Gini',
    trendPenalty: 'Downward-trend penalty',
    variationPenalty: 'Residual-variation penalty',
  };
  const components = Object.entries(labels).map(([key, label]) => ({
    key,
    label,
    baseline: baseline.metric[key],
    candidate: candidate.metric[key],
    delta: delta[key],
  }));
  const weekly = baseline.weekly.map((row, index) => {
    const candidateRow = candidate.weekly[index];
    const baselineTrend =
      baseline.metric.intercept + baseline.metric.slope * index;
    const candidateTrend =
      candidate.metric.intercept + candidate.metric.slope * index;
    return {
      week: row.week,
      baselineGini: row.gini,
      candidateGini: candidateRow.gini,
      deltaGini: candidateRow.gini - row.gini,
      baselineTrend,
      candidateTrend,
      deltaTrend: candidateTrend - baselineTrend,
    };
  });
  const tolerance = 1e-12;
  const checks = {
    sameWeeks:
      baseline.weekly.length === candidate.weekly.length &&
      baseline.weekly.every(
        (row, index) => row.week === candidate.weekly[index].week,
      ),
    weeklyDeltaZeroMean:
      Math.abs(
        weekly.reduce((sum, row) => sum + row.deltaGini, 0) / weekly.length,
      ) <= tolerance,
    scoreDeltaEqualsComponents:
      Math.abs(
        delta.stability -
          components.reduce((sum, component) => sum + component.delta, 0),
      ) <= tolerance,
  };
  if (!Object.values(checks).every(Boolean))
    throw new Error('Synthetic comparison invariants failed.');
  return {
    schema: 'home-credit-stability-comparison-v1',
    evidence_type: 'SYNTHETIC_ONLY',
    dataType:
      'Paired deterministic synthetic weekly Gini fixtures; not model inference or applicant data.',
    baseline,
    candidate,
    weekly,
    delta,
    components,
    checks,
    direction:
      Math.abs(delta.stability) <= tolerance
        ? 'unchanged'
        : delta.stability > 0
          ? 'higher'
          : 'lower',
    comparisonScope:
      'Candidate minus pinned baseline across the same 24 ordinal weeks. Both share mean Gini 0.6 and the same deterministic residual pattern; only decline and residual variation change. Higher synthetic stability is an illustration, not evidence of a better credit model or a release decision.',
  };
}

/** CSV retains paired weekly rows; JSON comparison additionally carries score components. */
export function comparisonCSV(settingsA = {}, settingsB = {}) {
  const comparison = compareFixtures(settingsA, settingsB);
  const header =
    'data_type,week,baseline_synthetic_weekly_gini,candidate_synthetic_weekly_gini,delta_gini,baseline_fitted_trend,candidate_fitted_trend,delta_fitted_trend';
  const fields = [
    'baselineGini',
    'candidateGini',
    'deltaGini',
    'baselineTrend',
    'candidateTrend',
    'deltaTrend',
  ];
  return (
    header +
    '\n' +
    comparison.weekly
      .map((row) =>
        [
          'synthetic_comparison',
          row.week,
          ...fields.map((key) => row[key].toFixed(10)),
        ].join(','),
      )
      .join('\n') +
    '\n'
  );
}
