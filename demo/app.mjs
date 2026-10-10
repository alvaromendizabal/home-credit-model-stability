import { SCENARIOS, makeFixture, stabilityMetric, reviewFixture, exportFixture, fixtureCSV } from './model.mjs';

const $ = id => document.getElementById(id);
const settings = { ...SCENARIOS.stable };
const svgNS = 'http://www.w3.org/2000/svg';
const format = (value, digits = 4) => (Math.abs(value) < 10 ** (-digits) / 2 ? 0 : value).toFixed(digits).replace('-', '−');

function svgElement(name, attributes, text) {
  const element = document.createElementNS(svgNS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderChart(rows, metric) {
  const group = $('chart-content');
  group.replaceChildren();
  const bounds = { left: 48, right: 702, top: 20, bottom: 278 };
  const lower = 0.35;
  const upper = 0.85;
  const x = index => bounds.left + index / (rows.length - 1) * (bounds.right - bounds.left);
  const y = value => bounds.bottom - (value - lower) / (upper - lower) * (bounds.bottom - bounds.top);
  const defs = svgElement('defs', {});
  const gradient = svgElement('linearGradient', { id: 'chart-fill', x1: 0, y1: 0, x2: 0, y2: 1 });
  gradient.append(svgElement('stop', { offset: '0%', 'stop-color': '#d9efdf', 'stop-opacity': '0.72' }), svgElement('stop', { offset: '100%', 'stop-color': '#edf6ee', 'stop-opacity': '0.05' }));
  defs.append(gradient);
  group.append(defs);
  [0.4, 0.5, 0.6, 0.7, 0.8].forEach(tick => {
    group.append(svgElement('line', { x1: bounds.left, x2: bounds.right, y1: y(tick), y2: y(tick), class: 'chart-grid' }));
    group.append(svgElement('text', { x: bounds.left - 12, y: y(tick) + 4, 'text-anchor': 'end', class: 'chart-axis-label' }, tick.toFixed(2)));
  });
  [0, 3, 7, 11, 15, 19, 23].forEach(i => group.append(svgElement('text', { x: x(i), y: 303, 'text-anchor': 'middle', class: 'chart-axis-label' }, `W${i + 1}`)));
  const points = rows.map((row, i) => `${x(i).toFixed(2)},${y(row.gini).toFixed(2)}`).join(' ');
  group.append(svgElement('polygon', { points: `${bounds.left},${bounds.bottom} ${points} ${bounds.right},${bounds.bottom}`, fill: 'url(#chart-fill)' }));
  group.append(svgElement('path', { d: `M${x(0)} ${y(metric.intercept)} L${x(rows.length - 1)} ${y(metric.intercept + metric.slope * (rows.length - 1))}`, class: 'chart-trend' }));
  group.append(svgElement('polyline', { points, class: 'chart-data' }));
  rows.forEach((row, i) => {
    const point = svgElement('circle', { cx: x(i), cy: y(row.gini), r: 2.8, class: 'chart-point' });
    point.append(svgElement('title', {}, `Week ${row.week}: Gini ${format(row.gini)}`));
    group.append(point);
  });
  $('chart-title').textContent = 'Synthetic weekly Gini across 24 consecutive weeks';
  $('chart-desc').textContent = `Synthetic fixture, not model predictions. Mean Gini ${format(metric.mean)}, fitted weekly slope ${format(metric.slope, 5)}, residual standard deviation ${format(metric.residualStd)}, stability score ${format(metric.stability)}. Weekly values and fitted trend are available in the data table.`;
}

function renderReview(metric) {
  const review = reviewFixture(metric);
  const card = document.querySelector('.review-card');
  card.classList.toggle('needs-review', !review.passed);
  $('review-status').textContent = review.passed ? 'Ready for review' : 'Investigate before review';
  $('review-icon').textContent = review.passed ? '✓' : '!';
  $('review-checks').replaceChildren(...review.checks.map(check => {
    const row = document.createElement('div');
    row.className = 'review-check';
    const label = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = check.label;
    const threshold = document.createElement('span');
    threshold.textContent = check.threshold;
    label.append(strong, threshold);
    const state = document.createElement('span');
    state.className = `check-state${check.passed ? '' : ' failed'}`;
    state.textContent = check.passed ? '✓ Meets example' : '↑ Outside example';
    row.append(label, state);
    return row;
  }));
}

function render() {
  const rows = makeFixture(settings);
  const metric = stabilityMetric(rows.map(row => row.gini));
  $('score').textContent = format(metric.stability);
  $('mean').textContent = format(metric.mean);
  $('trend').textContent = format(metric.trendPenalty);
  $('noise').textContent = format(metric.variationPenalty);
  $('decline-value').textContent = settings.decline.toFixed(5);
  $('variation-value').textContent = settings.variation.toFixed(3);
  ['decline', 'variation'].forEach(key => {
    $(key).value = settings[key];
    $(key).style.setProperty('--range-progress', `${100 * settings[key] / Number($(key).max)}%`);
    $(key).setAttribute('aria-valuetext', key === 'decline' ? `${settings.decline.toFixed(5)} Gini per week` : `${settings.variation.toFixed(3)} Gini residual standard deviation`);
  });
  document.querySelectorAll('[data-scenario]').forEach(button => {
    const preset = SCENARIOS[button.dataset.scenario];
    button.setAttribute('aria-pressed', String(Math.abs(preset.decline - settings.decline) < 1e-12 && Math.abs(preset.variation - settings.variation) < 1e-12));
  });
  const explanation = settings.decline >= 0.001 && settings.variation >= 0.04
    ? 'Both decline and inconsistency reduce the score. The average Gini stays at 0.6000, but the temporal profile calls for closer investigation.'
    : settings.decline >= 0.001
      ? 'A good average can conceal a declining model. The fitted downward trend receives a penalty even when weekly results are relatively consistent.'
      : settings.variation >= 0.04
        ? 'The average and fitted trend hide large week-to-week swings. Residual variation reduces the score and highlights inconsistent performance.'
        : settings.decline === 0 && settings.variation === 0
          ? 'With no decline or residual variation, the stability score equals the mean Gini. This perfectly flat profile is an illustrative boundary case.'
          : 'A steady profile preserves most of the average discrimination. Small declines still reduce its stability score.';
  $('interpretation').textContent = explanation;
  $('weekly-table').replaceChildren(...rows.map((row, i) => {
    const tr = document.createElement('tr');
    [row.week, format(row.gini, 6), format(metric.intercept + metric.slope * i, 6)].forEach(value => {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    });
    return tr;
  }));
  renderChart(rows, metric);
  renderReview(metric);
}

function download(formatName) {
  const isCSV = formatName === 'csv';
  const content = isCSV ? fixtureCSV(settings) : JSON.stringify(exportFixture(settings), null, 2) + '\n';
  const blob = new Blob([content], { type: isCSV ? 'text/csv;charset=utf-8' : 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `home-credit-synthetic-stability.${formatName}`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('export-status').textContent = `${formatName.toUpperCase()} download prepared with 24 synthetic weeks.`;
}

document.querySelectorAll('[data-scenario]').forEach(button => button.addEventListener('click', () => {
  Object.assign(settings, SCENARIOS[button.dataset.scenario]);
  render();
}));
['decline', 'variation'].forEach(key => $(key).addEventListener('input', event => {
  settings[key] = Number(event.target.value);
  render();
}));
$('reset').addEventListener('click', () => { Object.assign(settings, SCENARIOS.stable); render(); });
$('export-csv').addEventListener('click', () => download('csv'));
$('export-json').addEventListener('click', () => download('json'));
render();
document.querySelectorAll('button:disabled, input:disabled').forEach(element => { element.disabled = false; });
