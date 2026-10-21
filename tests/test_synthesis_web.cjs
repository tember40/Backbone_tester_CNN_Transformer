const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const math = require('../docs/assets/js/synthesis-math.js');
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

test('remote input changes full attention but not the local convolution output', () => {
  for (const [input, expected] of [[-2,[.4251946838417307,.7663259087188861]],[1,[.6169214179501338,.6169214179501338]],[2,[1.053904690937801,.5024838289508026]]]) {
    const result = math.mix(input);
    assert.deepEqual(result.convOutput, [.25,.75]);
    result.attentionOutput.forEach((value, i) => near(value, expected[i]));
    near(result.attentionWeights.reduce((a,b) => a+b,0), 1);
  }
  for (const invalid of [NaN, Infinity, -3, 3]) assert.throws(() => math.mix(invalid), RangeError);
});

test('confusion metrics distinguish overall accuracy, class recall and missing support', () => {
  near(math.summarizeMatrix(math.matrices.balanced).accuracy, 22/30*100);
  const skewed = math.summarizeMatrix(math.matrices.imbalanced);
  assert.equal(skewed.accuracy,90); near(skewed.macroRecall,100/3);
  assert.deepEqual(skewed.recalls,[100,0,0]);
  const missing = math.summarizeMatrix(math.matrices.missing);
  assert.deepEqual(missing.recalls,[80,90,null]); assert.equal(missing.macroRecall,85);
  assert.equal(math.summarizeMatrix([[0]]).accuracy,null);
  for (const invalid of [[],[[1,2]],[[NaN]],[[-1]],[[.5]]]) assert.throws(() => math.summarizeMatrix(invalid),TypeError);
});

test('curve fixtures stay within the fixed axis and use the first best accuracy epoch', () => {
  assert.equal(math.bestEpoch(math.curves.gap.accuracy),3);
  assert.equal(math.bestEpoch(math.curves.flat.accuracy),5);
  assert.equal(math.bestEpoch(math.curves.unstable.accuracy),5);
  for (const data of Object.values(math.curves)) {
    assert.equal(data.train.length,data.val.length);
    assert.ok([...data.train,...data.val].every(value => value>=0 && value<=4));
  }
});

function controller(ids, values) {
  const nodes = Object.fromEntries(ids.split(' ').map(id => {
    const handlers={}, attributes={};
    return [id,{value:values[id]??'',textContent:'',innerHTML:'',
      addEventListener(type,fn) {handlers[type]=fn;},emit(type) {handlers[type]?.();},
      setAttribute(key,value) {attributes[key]=value;},getAttribute:key=>attributes[key]}];
  }));
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../docs/assets/js/synthesis.js'),'utf8'),
    {window:{SynthesisMath:math},document:{querySelector:s=>nodes[s.slice(1)]??null,querySelectorAll:()=>[]},IntersectionObserver:class {observe() {}}});
  return nodes;
}

test('actual mixing controller clamps bad input and resets the numerical example', () => {
  const n=controller('mixFarValue mixInputValue mixVectors mixWeights mixConvOutput mixAttentionOutput mixExplanation mixReset',{mixFarValue:'1'});
  assert.equal(n.mixAttentionOutput.textContent,'[0.617, 0.617]');
  n.mixFarValue.value='2'; n.mixFarValue.emit('input');
  assert.equal(n.mixAttentionOutput.textContent,'[1.054, 0.502]');
  assert.equal(n.mixConvOutput.textContent,'[0.250, 0.750]');
  n.mixFarValue.value='999'; n.mixFarValue.emit('input'); assert.equal(n.mixFarValue.value,'2');
  n.mixFarValue.value='NaN'; n.mixFarValue.emit('input'); assert.equal(n.mixFarValue.value,'1');
  n.mixFarValue.value='-2'; n.mixFarValue.emit('input'); n.mixReset.emit('click');
  assert.equal(n.mixAttentionOutput.textContent,'[0.617, 0.617]');
});

test('actual failure controller updates curve, selected epoch and missing-class interpretation', () => {
  const n=controller('failureScenario failureTrainLine failureValLine failureCurveTicks failureCurveRows failureObservation failureHypothesis failureAction failureMatrixCase failureMatrixRows failureMatrixSummary failureClassRecall failureMatrixInterpretation',
    {failureScenario:'gap',failureMatrixCase:'balanced'});
  assert.match(n.failureObservation.textContent,/epoch 3/);
  assert.match(n.failureTrainLine.getAttribute('points'),/^60,140\.25/);
  n.failureScenario.value='flat'; n.failureScenario.emit('change');
  assert.match(n.failureObservation.textContent,/epoch 5/);
  assert.match(n.failureHypothesis.textContent,/gradient/);
  n.failureMatrixCase.value='imbalanced'; n.failureMatrixCase.emit('change');
  assert.match(n.failureMatrixSummary.textContent,/90\/100 = 90.00%.*33.33%/);
  n.failureMatrixCase.value='missing'; n.failureMatrixCase.emit('change');
  assert.match(n.failureClassRecall.textContent,/C: support 0, recall N\/A/);
  assert.match(n.failureMatrixSummary.textContent,/macro recall: 85.00%/);
});
