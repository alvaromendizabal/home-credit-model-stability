/** Browser-local synthetic stability explorer. No persistence or remote inference. */
export async function startLab({
  document = globalThis.document,
  loadModel = () => import('./model.mjs'),
  Blob = globalThis.Blob,
  URL = globalThis.URL,
  setTimeout = globalThis.setTimeout,
} = {}) {
  const $ = (id) => document.getElementById(id);
  const controls = () => document.querySelectorAll('button, input');
  const format = (value, digits = 4) =>
    (Math.abs(value) < 10 ** -digits / 2 ? 0 : value)
      .toFixed(digits)
      .replace('-', '−');
  const signed = (value) => `${value > 0.00005 ? '+' : ''}${format(value)}`;
  const svgNS = 'http://www.w3.org/2000/svg';
  const element = (name, className, text) => {
    const node = document.createElement(name);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svgElement = (name, attributes, text) => {
    const node = document.createElementNS(svgNS, name);
    Object.entries(attributes).forEach(([key, value]) =>
      node.setAttribute(key, String(value)),
    );
    if (text !== undefined) node.textContent = text;
    return node;
  };
  let model,
    settings,
    baseline = null,
    current = null,
    comparison = null;
  function clearResults(message) {
    current = null;
    comparison = null;
    for (const id of ['score', 'mean', 'trend', 'noise'])
      $(id).textContent = '—';
    for (const id of [
      'chart-content',
      'weekly-table',
      'review-checks',
      'delta-components',
    ])
      $(id).replaceChildren();
    $('chart-title').textContent = 'Synthetic chart unavailable';
    $('chart-desc').textContent =
      'Correct the controls or reset the candidate to calculate a current synthetic profile.';
    $('interpretation').textContent =
      'No current calculation. Correct the controls or reset the candidate.';
    $('review-status').textContent = 'Not calculated';
    $('review-icon').textContent = '—';
    document.querySelector('.review-card').classList.remove('needs-review');
    $('comparison-panel').hidden = true;
    $('baseline-legend').hidden = true;
    renderComparison();
    $('score-delta').textContent = '—';
    $('export-status').textContent = '';
    for (const id of ['export-json', 'export-csv', 'pin-baseline'])
      $(id).disabled = true;
    $('explorer-error').textContent = message;
    $('explorer-error').hidden = false;
  }
  function copySettings(source) {
    return { decline: source.decline, variation: source.variation };
  }
  function renderChart() {
    const rows = current.weekly,
      metric = current.metric;
    const group = $('chart-content');
    group.replaceChildren();
    const bounds = { left: 48, right: 702, top: 20, bottom: 278 };
    const lower = 0.35,
      upper = 0.85;
    const x = (index) =>
      bounds.left + (index / (rows.length - 1)) * (bounds.right - bounds.left);
    const y = (value) =>
      bounds.bottom -
      ((value - lower) / (upper - lower)) * (bounds.bottom - bounds.top);
    const defs = svgElement('defs', {});
    const gradient = svgElement('linearGradient', {
      id: 'chart-fill',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: 1,
    });
    gradient.append(
      svgElement('stop', {
        offset: '0%',
        'stop-color': '#d9efdf',
        'stop-opacity': '0.72',
      }),
      svgElement('stop', {
        offset: '100%',
        'stop-color': '#edf6ee',
        'stop-opacity': '0.05',
      }),
    );
    defs.append(gradient);
    group.append(defs);
    for (const tick of [0.4, 0.5, 0.6, 0.7, 0.8]) {
      group.append(
        svgElement('line', {
          x1: bounds.left,
          x2: bounds.right,
          y1: y(tick),
          y2: y(tick),
          class: 'chart-grid',
        }),
      );
      group.append(
        svgElement(
          'text',
          {
            x: bounds.left - 12,
            y: y(tick) + 4,
            'text-anchor': 'end',
            class: 'chart-axis-label',
          },
          tick.toFixed(2),
        ),
      );
    }
    for (const index of [0, 3, 7, 11, 15, 19, 23])
      group.append(
        svgElement(
          'text',
          {
            x: x(index),
            y: 303,
            'text-anchor': 'middle',
            class: 'chart-axis-label',
          },
          `W${rows[index].week}`,
        ),
      );
    const points = (values) =>
      values
        .map((row, index) => `${x(index).toFixed(2)},${y(row.gini).toFixed(2)}`)
        .join(' ');
    const candidatePoints = points(rows);
    group.append(
      svgElement('polygon', {
        points: `${bounds.left},${bounds.bottom} ${candidatePoints} ${bounds.right},${bounds.bottom}`,
        fill: 'url(#chart-fill)',
      }),
    );
    if (comparison)
      group.append(
        svgElement('polyline', {
          points: points(comparison.baseline.weekly),
          class: 'chart-baseline',
        }),
      );
    group.append(
      svgElement('path', {
        d: `M${x(0)} ${y(metric.intercept)} L${x(rows.length - 1)} ${y(metric.intercept + metric.slope * (rows.length - 1))}`,
        class: 'chart-trend',
      }),
    );
    group.append(
      svgElement('polyline', { points: candidatePoints, class: 'chart-data' }),
    );
    rows.forEach((row, index) => {
      const point = svgElement('circle', {
        cx: x(index),
        cy: y(row.gini),
        r: 2.8,
        class: 'chart-point',
      });
      point.append(
        svgElement(
          'title',
          {},
          `Week ${row.week}: candidate Gini ${format(row.gini)}${comparison ? `, baseline ${format(comparison.weekly[index].baselineGini)}` : ''}`,
        ),
      );
      group.append(point);
    });
    $('chart-title').textContent = comparison
      ? 'Synthetic candidate and pinned baseline on the same 24 weeks'
      : 'Synthetic weekly Gini across 24 consecutive weeks';
    $('chart-desc').textContent =
      `Synthetic fixture, not model predictions. Candidate mean Gini ${format(metric.mean)}, fitted weekly slope ${format(metric.slope, 5)}, residual standard deviation ${format(metric.residualStd)}, stability score ${format(metric.stability)}.${comparison ? ` Pinned baseline stability ${format(comparison.baseline.metric.stability)}; candidate minus baseline ${signed(comparison.delta.stability)}.` : ''} Weekly values are in the table; JSON exports preserve full precision.`;
    $('baseline-legend').hidden = !comparison;
  }
  function renderReview() {
    const review = current.illustrativeReview;
    document
      .querySelector('.review-card')
      .classList.toggle('needs-review', !review.passed);
    $('review-status').textContent = review.passed
      ? 'Ready for review'
      : 'Investigate before review';
    $('review-icon').textContent = review.passed ? '✓' : '!';
    $('review-checks').replaceChildren(
      ...review.checks.map((check) => {
        const row = element('div', 'review-check');
        const label = element('div');
        label.append(
          element('strong', '', check.label),
          element('span', '', check.threshold),
        );
        row.append(
          label,
          element(
            'span',
            `check-state${check.passed ? '' : ' failed'}`,
            check.passed ? '✓ Meets example' : '↑ Outside example',
          ),
        );
        return row;
      }),
    );
  }
  function renderComparison() {
    $('comparison-panel').hidden = !comparison;
    $('clear-baseline').hidden = !baseline;
    $('clear-baseline').disabled = !baseline;
    $('pin-baseline').textContent = baseline
      ? 'Replace pinned baseline'
      : 'Pin current baseline';
    $('reset').title = baseline
      ? 'Restore the candidate to the pinned baseline settings'
      : 'Restore the steady candidate settings';
    $('baseline-note').textContent = baseline
      ? `Pinned: decline ${baseline.decline.toFixed(5)}, variation ${baseline.variation.toFixed(3)}. Edit the candidate freely; this baseline stays fixed. Reset candidate returns to these settings.`
      : 'Pin a scenario, then change the candidate to compare the same 24 weeks. Nothing is saved after you leave this page.';
    if (!comparison) return;
    $('score-delta').textContent = signed(comparison.delta.stability);
    $('score-delta').className =
      comparison.direction === 'lower' ? 'lower' : '';
    $('baseline-score').textContent =
      `Pinned score ${format(comparison.baseline.metric.stability)}`;
    $('comparison-summary').textContent =
      comparison.direction === 'unchanged'
        ? 'The candidate and baseline have the same stability score at numerical precision.'
        : `The candidate’s synthetic stability score is ${comparison.direction} than the pinned baseline. All 24 weeks are aligned.`;
    $('delta-components').replaceChildren(
      ...comparison.components.map((component) => {
        const block = element('div', 'delta-component');
        block.append(
          element('span', '', component.label),
          element('strong', '', signed(component.delta)),
          element(
            'small',
            '',
            `${format(component.baseline)} → ${format(component.candidate)}`,
          ),
        );
        return block;
      }),
    );
  }
  function renderTable() {
    const headings = comparison
      ? ['Week', 'Baseline Gini', 'Candidate Gini', 'Candidate − baseline']
      : ['Week', 'Candidate Gini', 'Candidate fitted trend'];
    $('weekly-head').replaceChildren(
      ...headings.map((label) => {
        const th = element('th', '', label);
        th.setAttribute('scope', 'col');
        return th;
      }),
    );
    $('weekly-caption').textContent = comparison
      ? 'Same-week synthetic comparison. Both fitted trends are included in the exports.'
      : 'Synthetic weekly data and fitted trend';
    const rows = comparison
      ? comparison.weekly.map((row) => [
          row.week,
          format(row.baselineGini, 6),
          format(row.candidateGini, 6),
          format(row.deltaGini, 6),
        ])
      : current.weekly.map((row, index) => [
          row.week,
          format(row.gini, 6),
          format(current.metric.intercept + current.metric.slope * index, 6),
        ]);
    $('weekly-table').replaceChildren(
      ...rows.map((values) => {
        const tr = element('tr');
        tr.append(...values.map((value) => element('td', '', value)));
        return tr;
      }),
    );
  }
  function render() {
    try {
      const next = model.exportFixture(settings);
      const nextComparison = baseline
        ? model.compareFixtures(baseline, settings)
        : null;
      current = next;
      comparison = nextComparison;
      const metric = current.metric;
      $('score').textContent = format(metric.stability);
      $('mean').textContent = format(metric.mean);
      $('trend').textContent = format(metric.trendPenalty);
      $('noise').textContent = format(metric.variationPenalty);
      $('decline-value').textContent = settings.decline.toFixed(5);
      $('variation-value').textContent = settings.variation.toFixed(3);
      for (const key of ['decline', 'variation']) {
        $(key).value = settings[key];
        $(key).style.setProperty(
          '--range-progress',
          `${(100 * settings[key]) / Number($(key).max)}%`,
        );
        $(key).setAttribute(
          'aria-valuetext',
          key === 'decline'
            ? `${settings.decline.toFixed(5)} Gini per week`
            : `${settings.variation.toFixed(3)} Gini residual standard deviation`,
        );
      }
      document.querySelectorAll('[data-scenario]').forEach((button) => {
        const preset = model.SCENARIOS[button.dataset.scenario];
        button.setAttribute(
          'aria-pressed',
          String(
            Math.abs(preset.decline - settings.decline) < 1e-12 &&
              Math.abs(preset.variation - settings.variation) < 1e-12,
          ),
        );
      });
      $('interpretation').textContent =
        settings.decline >= 0.001 && settings.variation >= 0.04
          ? 'Both decline and inconsistency reduce the score. The average Gini stays at 0.6000, but the temporal profile calls for closer investigation.'
          : settings.decline >= 0.001
            ? 'A good average can conceal a declining model. The fitted downward trend receives a penalty even when weekly results are relatively consistent.'
            : settings.variation >= 0.04
              ? 'The average and fitted trend hide large week-to-week swings. Residual variation reduces the score and highlights inconsistent performance.'
              : settings.decline === 0 && settings.variation === 0
                ? 'With no decline or residual variation, the stability score equals the mean Gini. This perfectly flat profile is an illustrative boundary case.'
                : 'A steady profile preserves most of the average discrimination. Small declines still reduce its stability score.';
      renderChart();
      renderTable();
      renderReview();
      renderComparison();
      $('explorer-error').hidden = true;
      $('startup-status').hidden = true;
      $('export-status').textContent = '';
      for (const id of ['export-json', 'export-csv', 'pin-baseline'])
        $(id).disabled = false;
      return true;
    } catch (error) {
      clearResults(
        `Unable to calculate the candidate. ${error.message} Correct the controls or reset the candidate; no previous result is available for export.`,
      );
      return false;
    }
  }
  function download(formatName) {
    if (!current || $('export-json').disabled || $('export-csv').disabled)
      return;
    try {
      const isCSV = formatName === 'csv';
      const content = isCSV
        ? comparison
          ? model.comparisonCSV(
              comparison.baseline.parameters,
              comparison.candidate.parameters,
            )
          : model.fixtureCSV(current.parameters)
        : JSON.stringify(comparison || current, null, 2) + '\n';
      const blob = new Blob([content], {
        type: isCSV ? 'text/csv;charset=utf-8' : 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = element('a');
      link.href = url;
      link.download = `home-credit-synthetic-${comparison ? 'comparison' : 'stability'}.${formatName}`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      $('export-status').textContent =
        `${formatName.toUpperCase()} prepared: ${comparison ? 'baseline and candidate aligned on' : 'one illustrative profile across'} 24 synthetic weeks.`;
    } catch (error) {
      $('export-status').textContent =
        `Download unavailable: ${error.message}. Current calculations remain visible.`;
    }
  }
  try {
    model = await loadModel();
    settings = copySettings(model.SCENARIOS.stable);
    if (!render()) return false;
    document.querySelectorAll('[data-scenario]').forEach((button) =>
      button.addEventListener('click', () => {
        const preset = model.SCENARIOS[button.dataset.scenario];
        if (!preset) {
          clearResults('Select a valid synthetic scenario.');
          return;
        }
        settings = copySettings(preset);
        render();
      }),
    );
    for (const key of ['decline', 'variation'])
      $(key).addEventListener('input', (event) => {
        const raw = String(event.target.value).trim();
        settings[key] = raw === '' ? NaN : Number(raw);
        render();
      });
    $('reset').addEventListener('click', () => {
      settings = copySettings(baseline || model.SCENARIOS.stable);
      render();
    });
    $('pin-baseline').addEventListener('click', () => {
      if (!current) return;
      baseline = copySettings(current.parameters);
      render();
    });
    $('clear-baseline').addEventListener('click', () => {
      baseline = null;
      render();
      // The clear button becomes hidden. Move keyboard focus to its visible peer.
      if (document.activeElement === $('clear-baseline')) {
        ($('pin-baseline').disabled ? $('reset') : $('pin-baseline')).focus();
      }
    });
    $('export-csv').addEventListener('click', () => download('csv'));
    $('export-json').addEventListener('click', () => download('json'));
    controls().forEach((control) => {
      control.disabled = false;
    });
    $('clear-baseline').disabled = !baseline;
    return true;
  } catch (error) {
    controls().forEach((control) => {
      control.disabled = true;
    });
    clearResults(
      `The local explorer could not load. ${error.message} Reload the complete demo to try again.`,
    );
    $('startup-status').hidden = true;
    return false;
  }
}

if (typeof document !== 'undefined') await startLab();
