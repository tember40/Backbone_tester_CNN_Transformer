const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const math = require('../docs/assets/js/foundations-math.js');

const features = [[0, 0], [0, 1], [1, 0], [1, 1]];
const gates = { AND: [0, 0, 0, 1], OR: [0, 1, 1, 1], NAND: [1, 1, 1, 0], XOR: [0, 1, 1, 0] };
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test('zero score predicts one, then a zero-input error updates only the bias', () => {
  const weights = [0, 0], sample = [0, 0];
  const result = math.perceptronStep(weights, 0, sample, 0, 0.1);
  assert.equal(result.prediction, 1);
  assert.equal(result.error, -1);
  close(result.deltaBias, -0.1);
  assert.deepEqual(result.weightsAfter, [0, 0]);
  assert.equal(result.predictionAfter, 0);
  assert.deepEqual(weights, [0, 0]);
  assert.deepEqual(sample, [0, 0]);
});

test('AND first epoch follows the Python implementation sample order', () => {
  let weights = [0, 0], bias = 0;
  const errors = [], predictions = [];
  features.forEach((sample, index) => {
    const result = math.perceptronStep(weights, bias, sample, gates.AND[index], 0.1);
    errors.push(result.error); predictions.push(result.prediction);
    weights = result.weightsAfter; bias = result.biasAfter;
  });
  assert.deepEqual(predictions, [1, 0, 0, 0]);
  assert.deepEqual(errors, [-1, 0, 0, 1]);
  weights.forEach(value => close(value, 0.1));
  close(bias, 0);
});

test('separable gates converge; XOR cannot reach four correct predictions', () => {
  for (const [name, targets] of Object.entries(gates)) {
    let weights = [0, 0], bias = 0;
    for (let step = 0; step < 80; step++) {
      const i = step % 4;
      const result = math.perceptronStep(weights, bias, features[i], targets[i], 0.1);
      weights = result.weightsAfter; bias = result.biasAfter;
      if (name === 'XOR') assert.ok(features.some((x, j) => Number(x[0]*weights[0]+x[1]*weights[1]+bias >= 0) !== targets[j]));
    }
    if (name !== 'XOR') assert.deepEqual(features.map(x => Number(x[0]*weights[0]+x[1]*weights[1]+bias >= 0)), targets);
  }
});

test('invalid arithmetic inputs fail instead of leaking NaN into the drawing', () => {
  for (const args of [[[0], 0, [1, 0], 1, 0.1], [[0], NaN, [1], 1, 0.1], [[0], 0, [1], 2, 0.1], [[0], 0, [1], 1, 0]]) {
    assert.throws(() => math.perceptronStep(...args), RangeError);
  }
});

// Minimal DOM adapter: execute the real lesson controllers, not a second trainer.
// Layout and browser rendering are verified separately in a real browser.
function lesson(script, ids, values = {}) {
  const timers = new Map(); let nextTimer = 0;
  const context = new Proxy({}, { get: (_, key) => key === 'measureText' ? () => ({width: 10}) : () => {} });
  const nodes = Object.fromEntries(ids.split(' ').map(id => {
    const handlers = {};
    return [id, {
      value: values[id] ?? '', textContent: '', innerHTML: '', disabled: false,
      width: 600, height: 480, style: {}, classList: {toggle() {}},
      getContext: () => context,
      addEventListener(type, handler) { handlers[type] = handler; },
      click() { if (!this.disabled) handlers.click?.(); },
      change() { handlers.change?.(); },
    }];
  }));
  const sandbox = {
    document: {querySelector: selector => nodes[selector.slice(1)] ?? null, querySelectorAll: () => []},
    window: {setInterval(fn) { const id = ++nextTimer; timers.set(id, fn); return id; }, clearInterval(id) {timers.delete(id);}},
    BackboneFoundations: math,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../docs/assets/js', script), 'utf8'), sandbox);
  return {nodes, timers, tick() { for (const callback of [...timers.values()]) callback(); }};
}

test('perceptron UI shows before/after arithmetic and reset clears the trace and timer', () => {
  const ui = lesson('perceptron.js', 'boundaryCanvas gateSelect learningRate epochCount resetTraining stepTraining runTraining statusEpoch statusSample statusPrediction statusError statusWeights statusBias statusAccuracy updateExplanation truthTable trainingLog boundaryLabel stepTraceSummary stepTraceRows', {gateSelect: 'AND', learningRate: '0.1', epochCount: '5'});
  const n = ui.nodes;
  n.stepTraining.click();
  assert.match(n.stepTraceSummary.textContent, /e=0−1=-1/);
  assert.match(n.stepTraceRows.innerHTML, /−0.10/);
  assert.equal(n.statusBias.textContent, '−0.10');
  assert.equal(n.statusPrediction.textContent, '1 / 0');
  n.runTraining.click(); n.runTraining.click();
  assert.equal(ui.timers.size, 0);
  n.runTraining.click(); n.resetTraining.click();
  assert.equal(ui.timers.size, 0);
  assert.equal(n.statusEpoch.textContent, '0 / 5');
  assert.match(n.stepTraceRows.innerHTML, /아직 처리하지/);
});

test('MLP auto mode isolates manual steps, resets reproducibly, and reports the actual stop reason', () => {
  const ui = lesson('multilayer-perceptron.js', 'mlpBoundaryCanvas mlpLossCanvas mlpLearningRate mlpReset mlpStep mlpHundred mlpAuto mlpEpoch mlpLoss mlpAccuracy mlpTruthResults mlpStatusText', {mlpLearningRate: '0.1'});
  const n = ui.nodes;
  const initialLoss = n.mlpLoss.textContent;
  n.mlpAuto.click();
  assert.equal(n.mlpStep.disabled, true);
  assert.equal(n.mlpHundred.disabled, true);
  n.mlpStep.click();
  assert.equal(n.mlpEpoch.textContent, 0);
  for (let tick = 0; tick < 200 && ui.timers.size; tick++) ui.tick();
  assert.equal(ui.timers.size, 0);
  assert.equal(n.mlpStep.disabled, false);
  if (Number(n.mlpLoss.textContent) >= 0.03 || n.mlpAccuracy.textContent !== '100%') {
    assert.equal(n.mlpEpoch.textContent, 2000);
    assert.match(n.mlpStatusText.textContent, /목표 손실에 도달하지/);
  } else assert.match(n.mlpStatusText.textContent, /0.03 미만/);
  n.mlpReset.click();
  assert.equal(n.mlpEpoch.textContent, 0);
  assert.equal(n.mlpLoss.textContent, initialLoss);
});
