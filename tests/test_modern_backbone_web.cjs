const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Run the real controllers, providing only DOM nodes/events (not their calculations).
function lesson(script, ids, groups = {}) {
  let activeElement = null;
  function node() {
    const handlers = {};
    const classes = new Set();
    const attributes = {};
    return {dataset:{}, children:[], textContent:'', innerHTML:'', tabIndex:0,
      style:{setProperty() {}},
      classList:{toggle(name, on) {on ? classes.add(name) : classes.delete(name);}, contains:n => classes.has(n)},
      setAttribute(name, value) {attributes[name] = value;}, getAttribute:name => attributes[name],
      addEventListener(type, handler) {handlers[type] = handler;},
      emit(type, event = {}) {handlers[type]?.(event);},
      replaceChildren(...children) {this.children = children;}, append(child) {this.children.push(child);},
      focus() {activeElement = this;},
      getContext() {return null;}, getBoundingClientRect() {return {left:0, top:0, width:320, height:320};},
    };
  }
  const nodes = Object.fromEntries(ids.split(' ').map(id => [id, node()]));
  const controls = Object.fromEntries(Object.entries(groups).map(([selector, values]) =>
    [selector, values.map(dataset => Object.assign(node(), {dataset}))]));
  const document = {querySelector:selector => nodes[selector.slice(1)] ?? null,
    querySelectorAll:selector => controls[selector] ?? [], createElement:node};
  Object.defineProperty(document, 'activeElement', {get:() => activeElement});
  const sandbox = {document, IntersectionObserver:class {observe() {}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../docs/assets/js', script), 'utf8'), sandbox);
  return {nodes, controls, document};
}

test('Attention query buttons connect scores, stable normalization and the Value sum', () => {
  const {nodes:n, controls:c} = lesson('attention-basics.js',
    'attentionMatrix attentionScores attentionWeights attentionWeightSum attentionNormalization attentionOutput attentionBreakdown',
    {'[data-query]':[{query:'0'}, {query:'1'}, {query:'2'}]});
  assert.equal(n.attentionScores.textContent, '[0.71, 0.00, 0.71]');
  assert.equal(n.attentionWeights.textContent, '[0.401, 0.198, 0.401]');
  assert.equal(n.attentionOutput.textContent, '[0.802, 0.599]');
  assert.match(n.attentionNormalization.textContent, /\[1.000000, 0.493069, 1.000000\].*2.493069/);
  c['[data-query]'][2].emit('click');
  assert.equal(n.attentionWeights.textContent, '[0.248, 0.248, 0.503]');
  assert.equal(n.attentionOutput.textContent, '[0.752, 0.752]');
  assert.match(n.attentionNormalization.textContent, /1.986137/);
  assert.equal(c['[data-query]'][0].getAttribute('aria-pressed'), 'false');
  c['[data-query]'][1].emit('click');
  assert.equal(n.attentionOutput.textContent, '[0.599, 0.802]');
  c['[data-query]'][0].emit('click');
  assert.equal(n.attentionWeightSum.textContent, 'Query A 행의 합 = 1.000');
});

test('ViT patch sizes and bounded keyboard movement update the actual controller', () => {
  const {nodes:n, controls:c} = lesson('vision-transformer.js',
    'vitImage vitPatchPreview vitPatchCount vitPatchVector vitTokenCount vitAttentionCells vitPatchCoordinate vitPatchRange vitPatchFlatten vitPoolExplanation',
    {'[data-patch-size]':[{patchSize:'2'}, {patchSize:'4'}, {patchSize:'8'}, {patchSize:'16'}],
      '[data-pool]':[{pool:'cls'}, {pool:'mean'}]});
  assert.equal(n.vitTokenCount.textContent, '65');
  assert.equal(n.vitPatchRange.textContent, 'x=12–15, y=12–15');
  const expected = [['256','12','257','66,049'], ['64','48','65','4,225'],
    ['16','192','17','289'], ['4','768','5','25']];
  c['[data-patch-size]'].forEach((button, index) => {
    button.emit('click');
    assert.deepEqual([n.vitPatchCount.textContent,n.vitPatchVector.textContent,n.vitTokenCount.textContent,n.vitAttentionCells.textContent], expected[index]);
  });
  c['[data-pool]'][1].emit('click');
  assert.match(n.vitPoolExplanation.innerHTML, /65개 토큰 전체/); // Deliberately fixed P=4 pooling example.
  c['[data-pool]'][0].emit('click');
  assert.match(n.vitPoolExplanation.innerHTML, /x\[:, 0\]/);
  n.vitImage.emit('keydown', {key:'ArrowRight', preventDefault() {}});
  assert.equal(n.vitPatchRange.textContent, 'x=16–31, y=0–15');
  n.vitImage.emit('keydown', {key:'ArrowRight', preventDefault() {}});
  assert.equal(n.vitPatchRange.textContent, 'x=16–31, y=0–15');
  n.vitImage.emit('click', {clientX:-5, clientY:999});
  assert.equal(n.vitPatchRange.textContent, 'x=0–15, y=16–31');
});

test('PVT modes retain queries while changing K/V and warn about a single key', () => {
  const {nodes:n, controls:c} = lesson('pvt.js',
    'pvtStageOne pvtStageTwo pvtStageThree pvtStageFour pvtModeLabel pvtStageTitle pvtStageDescription pvtStageStructure pvtStageRatio pvtQLabel pvtKVLabel pvtQueryCount pvtKeyCount pvtReducedCells pvtFullCells pvtStageWarning pvtQGrid pvtKVGrid',
    {'[data-pvt-mode]':[{pvtMode:'cifar'}, {pvtMode:'naive'}, {pvtMode:'paper'}],
      '[data-pvt-stage]':[0,1,2,3].map(i => ({pvtStage:String(i)}))});
  assert.equal(n.pvtReducedCells.textContent, '1,024');
  assert.equal(n.pvtKeyCount.textContent, '16');
  c['[data-pvt-mode]'][1].emit('click');
  assert.equal(n.pvtQueryCount.textContent, '64');
  assert.equal(n.pvtKeyCount.textContent, '1');
  assert.equal(n.pvtStageWarning.classList.contains('is-warning'), true);
  assert.match(n.pvtStageWarning.textContent, /softmax 가중치는 1/);
  c['[data-pvt-mode]'][2].emit('click');
  assert.equal(n.pvtQueryCount.textContent, '3,136');
  assert.equal(n.pvtKeyCount.textContent, '49');
  assert.equal(n.pvtQGrid.children.length, 64); // Display capped at 8×8, not the numerical count.
  c['[data-pvt-mode]'][0].emit('click');
  c['[data-pvt-stage]'][3].emit('click');
  assert.equal(n.pvtQueryCount.textContent, '1');
  assert.match(n.pvtStageWarning.textContent, /마지막 격자는 1×1/);
});

test('ConvNeXt grid keeps one Tab stop and restores focus after click and arrow navigation', () => {
  const {nodes:n, controls:c, document} = lesson('convnext.js',
    'convnextGrid convnextWindow convnextStageTitle convnextStageInfo convnextKernelLabel convnextPosition convnextActive convnextWeights convnextCoverage convnextExplanation',
    {'[data-convnext-stage]':[0,1,2,3].map(i => ({convnextStage:String(i)})),
      '[data-convnext-kernel]':[{convnextKernel:'7'}, {convnextKernel:'3'}]});
  const selected = () => n.convnextGrid.children.filter(cell => cell.tabIndex === 0);
  assert.equal(n.convnextActive.textContent, '49');
  assert.equal(selected().length, 1);
  n.convnextGrid.children[0].emit('click');
  assert.equal(n.convnextActive.textContent, '16');
  assert.equal(n.convnextPosition.textContent, '1행 1열');
  assert.equal(document.activeElement, selected()[0]);
  assert.equal(document.activeElement.getAttribute('aria-pressed'), 'true');
  let prevented = false;
  selected()[0].emit('keydown', {key:'ArrowRight', preventDefault() {prevented = true;}});
  assert.equal(prevented, true);
  assert.equal(n.convnextPosition.textContent, '1행 2열');
  selected()[0].emit('keydown', {key:'End', preventDefault() {}});
  assert.equal(n.convnextPosition.textContent, '8행 8열');
  selected()[0].emit('keydown', {key:'ArrowRight', preventDefault() {}});
  assert.equal(n.convnextPosition.textContent, '8행 8열');
  selected()[0].emit('keydown', {key:'Home', preventDefault() {}});
  assert.equal(n.convnextPosition.textContent, '1행 1열');
  assert.equal(document.activeElement, selected()[0]);
  c['[data-convnext-stage]'][3].emit('click');
  assert.equal(n.convnextGrid.children.length, 1);
  assert.equal(n.convnextActive.textContent, '1');
  assert.equal(n.convnextCoverage.textContent, '2%');
  c['[data-convnext-kernel]'][1].emit('click');
  assert.equal(n.convnextWeights.textContent, '9');
  assert.equal(n.convnextCoverage.textContent, '11%');
});
