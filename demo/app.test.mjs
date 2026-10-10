/** Actual app handlers with the real deterministic engine and a dependency-free DOM. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as engine from './model.mjs';
import { startLab } from './app.mjs';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const css = readFileSync(new URL('./styles.css', import.meta.url), 'utf8');
const close = (a, b, tolerance = 1e-12) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const number = (text) => Number(text.replace('−', '-'));
const settings = (preset) => ({
  decline: preset.decline,
  variation: preset.variation,
});

async function browser({
  loadFailure = false,
  downloadFailure = false,
  model = engine,
} = {}) {
  const nodes = new Map(),
    all = [],
    scenarios = [],
    timers = [];
  let active = null,
    blob = null,
    download = null,
    revoked = null;
  class Element {
    constructor(tag = 'div') {
      this.tag = tag;
      this.children = [];
      this.attributes = {};
      this.handlers = {};
      this.dataset = {};
      this.style = {
        setProperty(name, value) {
          this[name] = String(value);
        },
      };
      this.hidden = false;
      this.disabled = false;
      this._text = '';
      this._value = '';
      this.className = '';
      this.classList = {
        toggle: (name, force) => {
          const values = new Set(this.className.split(/\s+/).filter(Boolean));
          if (force) values.add(name);
          else values.delete(name);
          this.className = [...values].join(' ');
        },
        remove: (name) => {
          this.className = this.className
            .split(/\s+/)
            .filter((value) => value !== name)
            .join(' ');
        },
      };
      all.push(this);
    }
    set id(value) {
      this._id = value;
      nodes.set(value, this);
    }
    get id() {
      return this._id;
    }
    set value(value) {
      this._value = String(value);
    }
    get value() {
      return this._value;
    }
    set textContent(value) {
      this._text = String(value);
      this.children = [];
    }
    get textContent() {
      return (
        this._text + this.children.map((child) => child.textContent).join('')
      );
    }
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    }
    append(...children) {
      this.children.push(...children);
    }
    replaceChildren(...children) {
      this._text = '';
      this.children = children;
    }
    addEventListener(name, callback) {
      this.handlers[name] = callback;
    }
    focus() {
      active = this;
    }
    click() {
      if (this.disabled) return;
      this.handlers.click?.({ target: this });
      if (this.download) download = this;
    }
    remove() {
      this.removed = true;
    }
  }
  for (const match of html.matchAll(
    /<([a-z]+)[^>]*(?:\bid="[^"]+"|\bdata-scenario="[^"]+")[^>]*>/g,
  )) {
    const element = new Element(match[1]);
    const id = match[0].match(/\bid="([^"]+)"/);
    const scenario = match[0].match(/data-scenario="([^"]+)"/);
    if (id) element.id = id[1];
    if (scenario) {
      element.dataset.scenario = scenario[1];
      scenarios.push(element);
    }
    element.hidden = /\bhidden(?:\s|>)/.test(match[0]);
    element.disabled = /\bdisabled(?:\s|>)/.test(match[0]);
    element.max = match[0].match(/\bmax="([^"]+)"/)?.[1];
    element.value = match[0].match(/\bvalue="([^"]+)"/)?.[1] ?? '';
  }
  const reviewCard = new Element();
  reviewCard.className = 'review-card';
  const document = {
    body: new Element('body'),
    get activeElement() {
      return active;
    },
    getElementById: (id) => nodes.get(id),
    createElement: (tag) => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    querySelector: (selector) => {
      assert.equal(selector, '.review-card');
      return reviewCard;
    },
    querySelectorAll: (selector) => {
      if (selector === '[data-scenario]') return scenarios;
      if (selector === 'button, input')
        return all.filter((node) => ['button', 'input'].includes(node.tag));
      throw new Error(`Unimplemented selector: ${selector}`);
    },
  };
  const ready = await startLab({
    document,
    loadModel: async () => {
      if (loadFailure) throw new Error('fixture module unavailable');
      return model;
    },
    Blob,
    URL: {
      createObjectURL(value) {
        if (downloadFailure) throw new Error('fixture download blocked');
        blob = value;
        return 'blob:synthetic';
      },
      revokeObjectURL(value) {
        revoked = value;
      },
    },
    setTimeout: (callback) => {
      timers.push(callback);
      return timers.length;
    },
  });
  const ui = {
    ready,
    document,
    nodes,
    all,
    el: (id) => {
      assert.ok(nodes.has(id), `exists: ${id}`);
      return nodes.get(id);
    },
    scenario: (name) => {
      const button = scenarios.find((node) => node.dataset.scenario === name);
      assert.ok(button);
      button.click();
    },
    click: (id) => ui.el(id).click(),
    input: (id, value) => {
      const control = ui.el(id);
      control.value = value;
      assert.equal(typeof control.handlers.input, 'function');
      control.handlers.input({ target: control });
    },
    async exportJSON() {
      ui.click('export-json');
      assert.ok(blob, 'JSON created');
      return JSON.parse(await blob.text());
    },
    async exportCSV() {
      ui.click('export-csv');
      assert.ok(blob, 'CSV created');
      return blob.text();
    },
    blob: () => blob,
    download: () => download,
    finishTimers: () => {
      for (const callback of timers.splice(0)) callback();
      return revoked;
    },
  };
  return ui;
}
const descendants = (node) => [node, ...node.children.flatMap(descendants)];
const paths = (ui, className) =>
  descendants(ui.el('chart-content')).filter(
    (node) => node.attributes.class === className,
  );

test('startup computes the real steady fixture and keeps comparison hidden until pinned', async () => {
  const ui = await browser();
  assert.equal(ui.ready, true);
  assert.equal(ui.el('score').textContent, '0.5710');
  assert.equal(ui.el('mean').textContent, '0.6000');
  assert.equal(ui.el('startup-status').hidden, true);
  assert.equal(ui.el('explorer-error').hidden, true);
  assert.equal(ui.el('comparison-panel').hidden, true);
  assert.equal(ui.el('weekly-table').children.length, 24);
  assert.equal(paths(ui, 'chart-data').length, 1);
  assert.equal(paths(ui, 'chart-baseline').length, 0);
  assert.equal(ui.el('clear-baseline').disabled, true);
  assert.deepEqual(
    await ui.exportJSON(),
    engine.exportFixture(settings(engine.SCENARIOS.stable)),
  );
});

test('pinning copies the current scenario and starts with zero aligned differences', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  const exported = await ui.exportJSON();
  assert.equal(exported.schema, 'home-credit-stability-comparison-v1');
  assert.equal(exported.evidence_type, 'SYNTHETIC_ONLY');
  assert.deepEqual(exported.baseline, exported.candidate);
  assert.equal(exported.direction, 'unchanged');
  assert.ok(Object.values(exported.checks).every(Boolean));
  assert.equal(ui.el('score-delta').textContent, '0.0000');
  assert.equal(ui.el('comparison-panel').hidden, false);
  assert.equal(ui.el('baseline-legend').hidden, false);
  assert.equal(paths(ui, 'chart-baseline').length, 1);
});

test('editing the candidate never mutates the pinned baseline and gives the actual score delta', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  const original = (await ui.exportJSON()).baseline;
  ui.input('decline', 0.002);
  const changed = await ui.exportJSON();
  assert.deepEqual(changed.baseline, original);
  close(changed.delta.stability, -0.154);
  assert.equal(ui.el('score-delta').textContent, '−0.1540');
  assert.match(ui.el('comparison-summary').textContent, /lower/);
  assert.equal(ui.el('delta-components').children.length, 3);
  changed.components.forEach((component, index) =>
    close(
      number(ui.el('delta-components').children[index].children[1].textContent),
      component.delta,
    ),
  );
});

test('candidate and baseline overlays share identical week coordinates and match exported values', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.scenario('volatile');
  const comparison = await ui.exportJSON();
  const baselinePoints = paths(ui, 'chart-baseline')[0]
    .attributes.points.split(' ')
    .map((point) => point.split(',').map(Number));
  const candidatePoints = paths(ui, 'chart-data')[0]
    .attributes.points.split(' ')
    .map((point) => point.split(',').map(Number));
  assert.equal(candidatePoints.length, 24);
  for (let i = 0; i < 24; i++) {
    close(baselinePoints[i][0], candidatePoints[i][0]);
    close(
      candidatePoints[i][1],
      278 - ((comparison.weekly[i].candidateGini - 0.35) / 0.5) * 258,
      0.0051,
    );
    close(
      baselinePoints[i][1],
      278 - ((comparison.weekly[i].baselineGini - 0.35) / 0.5) * 258,
      0.0051,
    );
  }
});

test('paired table uses all 24 aligned values with candidate-minus-baseline signs', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.scenario('drift');
  const result = await ui.exportJSON();
  assert.equal(ui.el('weekly-head').children.length, 4);
  result.weekly.forEach((row, index) => {
    const cells = ui.el('weekly-table').children[index].children;
    assert.equal(number(cells[0].textContent), row.week);
    close(number(cells[1].textContent), row.baselineGini, 0.00000051);
    close(number(cells[2].textContent), row.candidateGini, 0.00000051);
    close(number(cells[3].textContent), row.deltaGini, 0.00000051);
  });
});

test('changing both sliders attributes the full difference to the three actual score components', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.input('decline', 0.001);
  ui.input('variation', 0.05);
  const result = await ui.exportJSON();
  close(
    result.delta.stability,
    result.components.reduce((sum, component) => sum + component.delta, 0),
  );
  close(result.delta.mean, 0);
  close(result.delta.trendPenalty, -88 * (0.001 - 0.00025));
  close(result.delta.variationPenalty, -0.5 * (0.05 - 0.014));
});

test('reset candidate restores the pinned settings rather than erasing the baseline', async () => {
  const ui = await browser();
  ui.scenario('volatile');
  ui.click('pin-baseline');
  ui.scenario('drift');
  ui.click('reset');
  const result = await ui.exportJSON();
  assert.deepEqual(
    result.baseline.parameters,
    settings(engine.SCENARIOS.volatile),
  );
  assert.deepEqual(result.candidate.parameters, result.baseline.parameters);
  assert.equal(ui.el('comparison-panel').hidden, false);
  assert.equal(ui.el('score-delta').textContent, '0.0000');
});

test('replacing the baseline uses the current candidate and subsequent edits remain independent', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.scenario('drift');
  ui.click('pin-baseline');
  const repinned = await ui.exportJSON();
  assert.deepEqual(
    repinned.baseline.parameters,
    settings(engine.SCENARIOS.drift),
  );
  assert.equal(repinned.direction, 'unchanged');
  ui.scenario('stable');
  const changed = await ui.exportJSON();
  assert.deepEqual(changed.baseline, repinned.baseline);
  close(changed.delta.stability, 0.154);
  assert.equal(ui.el('score-delta').textContent, '+0.1540');
});

test('clear baseline retains the candidate, returns single exports, and transfers focus off the hidden control', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.scenario('drift');
  ui.el('clear-baseline').focus();
  ui.click('clear-baseline');
  assert.equal(ui.el('clear-baseline').hidden, true);
  assert.equal(ui.el('comparison-panel').hidden, true);
  assert.equal(ui.document.activeElement, ui.el('pin-baseline'));
  assert.equal(paths(ui, 'chart-baseline').length, 0);
  assert.equal(ui.el('weekly-head').children.length, 3);
  const result = await ui.exportJSON();
  assert.equal(result.schema, 'home-credit-stability-illustration-v1');
  assert.deepEqual(result.parameters, settings(engine.SCENARIOS.drift));
});

test('range and preset handlers retain keyboard focus while updating chart and review content', async () => {
  const ui = await browser();
  const slider = ui.el('decline');
  slider.focus();
  ui.input('decline', 0.002);
  assert.equal(ui.document.activeElement, slider);
  assert.match(slider.attributes['aria-valuetext'], /0.00200 Gini per week/);
  const pin = ui.el('pin-baseline');
  pin.focus();
  ui.click('pin-baseline');
  assert.equal(ui.document.activeElement, pin);
  assert.equal(ui.el('review-status').textContent, 'Investigate before review');
});

test('clearing a baseline during invalid candidate input clears its labels and keeps recovery keyboard-accessible', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.input('decline', 'NaN');
  ui.el('clear-baseline').focus();
  ui.click('clear-baseline');
  assert.equal(ui.el('clear-baseline').hidden, true);
  assert.equal(ui.el('clear-baseline').disabled, true);
  assert.match(ui.el('baseline-note').textContent, /^Pin a scenario/);
  assert.equal(ui.el('export-json').disabled, true);
  assert.equal(ui.document.activeElement, ui.el('reset'));
  ui.click('reset');
  assert.equal(ui.el('comparison-panel').hidden, true);
  assert.equal(ui.el('score').textContent, '0.5710');
});

test('invalid inputs clear all computed output and block stale pinning or downloads until reset', async () => {
  for (const [id, value] of [
    ['decline', ''],
    ['decline', 'NaN'],
    ['decline', 0.0031],
    ['variation', -1],
    ['variation', 0.091],
    ['variation', 'Infinity'],
  ]) {
    const ui = await browser();
    ui.click('pin-baseline');
    ui.input(id, value);
    assert.equal(ui.el('explorer-error').hidden, false, `${id} ${value}`);
    assert.equal(ui.el('score').textContent, '—');
    assert.equal(ui.el('chart-content').children.length, 0);
    assert.equal(ui.el('weekly-table').children.length, 0);
    assert.equal(ui.el('review-status').textContent, 'Not calculated');
    assert.equal(ui.el('comparison-panel').hidden, true);
    for (const control of ['export-json', 'export-csv', 'pin-baseline'])
      assert.equal(ui.el(control).disabled, true);
    ui.click('export-json');
    assert.equal(ui.blob(), null);
    ui.click('reset');
    assert.equal(ui.el('explorer-error').hidden, true);
    assert.equal(ui.el('score').textContent, '0.5710');
    assert.equal(ui.el('comparison-panel').hidden, false);
  }
});

test('a real calculation exception cannot leave previous scores or exports active', async () => {
  let calls = 0;
  const ui = await browser({
    model: {
      ...engine,
      exportFixture: (...args) => {
        calls++;
        if (calls === 2) throw new Error('fixture calculation failed');
        return engine.exportFixture(...args);
      },
    },
  });
  ui.input('decline', 0.002);
  assert.match(
    ui.el('explorer-error').textContent,
    /fixture calculation failed/,
  );
  assert.equal(ui.el('score').textContent, '—');
  assert.equal(ui.el('export-json').disabled, true);
  ui.click('reset');
  assert.equal(ui.el('export-json').disabled, false);
});

test('model module load failure leaves every control disabled and no static performance claim', async () => {
  const ui = await browser({ loadFailure: true });
  assert.equal(ui.ready, false);
  assert.match(ui.el('explorer-error').textContent, /could not load/);
  assert.equal(ui.el('score').textContent, '—');
  assert.equal(ui.el('review-status').textContent, 'Not calculated');
  for (const control of ui.document.querySelectorAll('button, input'))
    assert.equal(control.disabled, true);
  for (const id of ['score', 'mean', 'trend', 'noise'])
    assert.match(
      html,
      new RegExp(`<strong\\s+id="${id}"\\s*>\\s*—\\s*</strong\\s*>`),
    );
  assert.match(html, /id="review-status"[^>]*>\s*Not\s+yet\s+calculated/);
  assert.match(
    html,
    /id="startup-status"[^>]*>\s*Interactive\s+calculations\s+have\s+not\s+loaded/,
  );
});

test('paired JSON is exactly replayable and download resources are released', async () => {
  const ui = await browser();
  ui.input('variation', 0.03);
  ui.click('pin-baseline');
  ui.input('decline', 0.0015);
  const result = await ui.exportJSON();
  assert.deepEqual(
    result,
    engine.compareFixtures(
      result.baseline.parameters,
      result.candidate.parameters,
    ),
  );
  assert.equal(ui.download().download, 'home-credit-synthetic-comparison.json');
  assert.equal(ui.download().removed, true);
  assert.equal(ui.finishTimers(), 'blob:synthetic');
});

test('comparison CSV contains the same 24 paired weeks and clearly labeled synthetic rows', async () => {
  const ui = await browser();
  ui.click('pin-baseline');
  ui.scenario('volatile');
  const csv = await ui.exportCSV();
  assert.equal(
    csv,
    engine.comparisonCSV(
      settings(engine.SCENARIOS.stable),
      settings(engine.SCENARIOS.volatile),
    ),
  );
  const rows = csv.trim().split('\n');
  assert.equal(rows.length, 25);
  assert.match(
    rows[0],
    /baseline_synthetic_weekly_gini,candidate_synthetic_weekly_gini,delta_gini/,
  );
  rows
    .slice(1)
    .forEach((row, index) =>
      assert.ok(row.startsWith(`synthetic_comparison,${index + 1},`)),
    );
  assert.equal(ui.download().download, 'home-credit-synthetic-comparison.csv');
});

test('single CSV export remains available, and unpinned reset returns to the steady profile', async () => {
  const ui = await browser();
  ui.scenario('volatile');
  assert.equal(
    await ui.exportCSV(),
    engine.fixtureCSV(settings(engine.SCENARIOS.volatile)),
  );
  ui.click('reset');
  assert.deepEqual(
    (await ui.exportJSON()).parameters,
    settings(engine.SCENARIOS.stable),
  );
  assert.equal(ui.el('comparison-panel').hidden, true);
});

test('download errors are announced visibly while valid calculations remain intact', async () => {
  const ui = await browser({ downloadFailure: true });
  ui.click('export-json');
  assert.equal(ui.blob(), null);
  assert.match(ui.el('export-status').textContent, /Download unavailable/);
  assert.equal(ui.el('score').textContent, '0.5710');
  assert.equal(ui.el('explorer-error').hidden, true);
});

test('all legal corner scenarios fit the fixed chart range and retain numerical results', async () => {
  for (const decline of [0, 0.003])
    for (const variation of [0, 0.09]) {
      const ui = await browser();
      ui.input('decline', decline);
      ui.input('variation', variation);
      const fixture = await ui.exportJSON();
      assert.equal(ui.el('explorer-error').hidden, true);
      assert.ok(
        fixture.weekly.every((row) => row.gini >= 0.35 && row.gini <= 0.85),
      );
      assert.ok(Number.isFinite(fixture.metric.stability));
    }
});

test('hidden states, readable CSS, accessible controls and local assets are explicit', () => {
  assert.match(css, /\[hidden\]\s*\{\s*display\s*:\s*none\s*!important/);
  for (const match of css.matchAll(/font(?:-size)?\s*:\s*(\d+(?:\.\d+)?)px/g))
    assert.ok(Number(match[1]) >= 12, `font size ${match[1]}px`);
  assert.match(css, /prefers-reduced-motion/);
  assert.match(css, /focus-visible/);
  assert.match(css, /input\[type=["']?range["']?\]\s*\{\s*height\s*:\s*24px/);
  assert.match(html, /class="chart-scroll"\s+tabindex="0"\s+role="region"/);
  for (const id of ['decline', 'variation'])
    assert.match(html, new RegExp(`for="${id}"`));
  for (const match of html.matchAll(/<script[^>]*src="([^"]+)"/g))
    assert.ok(match[1].startsWith('./'));
  assert.match(html, /role="alert"/);
});
