const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const math = require('../docs/assets/js/cnn-math.js');

test('cross-correlation does not flip the kernel; padding and stride match hand calculations', () => {
  assert.deepEqual(math.correlate2D([[1,2,3],[4,5,6],[7,8,9]], [[1,0],[0,-1]]), [[-4,-4],[-4,-4]]);
  assert.equal(math.outputSize(6,3,2), 2);
  assert.equal(math.outputSize(32,3,1,0), 30);
  assert.equal(math.outputSize(32,3,1,1), 32);
  assert.equal(math.outputSize(7,3,1,0,2), 3);
  assert.equal(math.outputSize(5,7), 0);
  assert.throws(() => math.correlate2D([[1]], [[1,1],[1,1]]), RangeError);
  assert.throws(() => math.outputSize(8,2,0), RangeError);
});

test('RGB partial correlations combine to one output, with one output bias', () => {
  const patches = [[[1,2],[3,4]], [[0,1],[2,0]], [[1,1],[1,1]]];
  const filters = [[[1,0],[0,-1]], [[0,1],[1,0]], [[1,0],[0,1]]];
  const partials = patches.map((input, i) => math.correlate2D(input, filters[i])[0][0]);
  assert.deepEqual(partials, [-3,3,2]);
  assert.equal(partials.reduce((a,b) => a+b, 0) - 0.5, 1.5);
});

test('pooling coverage includes interior holes when stride exceeds the window', () => {
  const geometry = math.poolingGeometry(8,2,3);
  assert.equal(geometry.output, 3);
  assert.equal(geometry.trailing, 0);
  assert.deepEqual(geometry.axis, [true,true,false,true,true,false,true,true]);
  assert.equal(geometry.uncovered, 28);
  assert.equal(math.poolingGeometry(5,2,2).uncovered, 9);
  assert.equal(math.poolingGeometry(8,3,1).uncovered, 0);
});

test('an aligned edge pools identically; a one-pixel shift makes four cells differ', () => {
  const aligned = Array.from({length:8}, () => Array.from({length:8}, (_, i) => i < 4 ? 0.15 : 0.9));
  const shifted = Array.from({length:8}, () => Array.from({length:8}, (_, i) => i < 3 ? 0.15 : 0.9));
  assert.deepEqual(math.pool2D(aligned), math.pool2D(aligned,2,2,'average'));
  const maximum = math.pool2D(shifted), average = math.pool2D(shifted,2,2,'average');
  assert.ok(Math.abs(average[0][1]-0.525)<1e-12);
  assert.equal(maximum[0][1],0.9);
  assert.equal(maximum.flat().filter((v,i) => Math.abs(v-average.flat()[i])>1e-12).length,4);
});

function lesson(script, ids, values = {}) {
  const timers = new Map(); let timerId = 0;
  const context = new Proxy({}, {get: () => () => {}});
  const nodes = Object.fromEntries(ids.split(' ').map(id => {
    const handlers = {};
    return [id, {value:values[id] ?? '', type:'range', textContent:'', innerHTML:'', disabled:false,
      width:256, height:256, style:{setProperty() {}}, classList:{toggle() {}},
      querySelectorAll:() => [],
      getContext:() => context,
      addEventListener(type, fn) {handlers[type]=fn;},
      emit(type) {handlers[type]?.();}, click() {handlers.click?.();},
    }];
  }));
  const sandbox = {document:{querySelector:s => nodes[s.slice(1)] ?? null, querySelectorAll:() => []},
    window:{setInterval(fn) {timers.set(++timerId,fn); return timerId;},clearInterval(id) {timers.delete(id);}},
    BackboneCNNMath:math, IntersectionObserver:class {observe() {}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../docs/assets/js',script),'utf8'),sandbox);
  return {nodes, timers};
}

test('shape calculator rejects impossible settings instead of displaying a negative image', () => {
  const {nodes:n} = lesson('convolution-filters.js','shapeInput shapeKernel shapePadding shapeStride shapeInputValue shapeInputLabel shapeOutput shapeSubstitution shapeInterpretation',
    {shapeInput:'5',shapeKernel:'7',shapePadding:'0',shapeStride:'1'});
  assert.equal(n.shapeOutput.textContent,'계산 불가');
  assert.match(n.shapeInterpretation.textContent,/커널 7칸이 큽니다/);
  assert.match(n.shapeSubstitution.textContent,/놓을 수 있는 창이 없습니다/);
  n.shapeKernel.value='3'; n.shapeKernel.emit('input');
  assert.equal(n.shapeOutput.textContent,'3 × 3');
});

test('pooling UI draws actual coverage and AlexNet all eight spatial operations', () => {
  const {nodes:n} = lesson('pooling-receptive-field.js','poolInputSize poolKernelSize poolStride poolInputSizeValue poolKernelSizeValue poolStrideValue poolGeometryFormula poolGeometryShape poolGeometryWindows poolGeometryRemainder poolGeometryGrid rfArchitecture rfStage rfStageValue rfLayerName rfSize rfJump rfFormula rfOverlay rfDisplayNote rfStageList',
    {poolInputSize:'8',poolKernelSize:'2',poolStride:'3',rfArchitecture:'alexnet',rfStage:'8'});
  assert.equal((n.poolGeometryGrid.innerHTML.match(/class="is-used"/g)||[]).length,36);
  assert.match(n.poolGeometryRemainder.textContent,/미사용 28\/64/);
  assert.equal(n.rfSize.textContent,'42 × 42');
  assert.equal(n.rfJump.textContent,'8px');
  assert.equal(n.rfStage.max,8);
  assert.equal(n.rfOverlay.style.width,'100%');
  assert.match(n.rfDisplayNote.textContent,/입력보다 큽니다/);
  n.rfStage.value='0'; n.rfStage.emit('input');
  assert.equal(n.rfOverlay.style.width,'3.125%');
});

test('unchanged positive activations use the same color before and after ReLU', () => {
  const {nodes:n} = lesson('feature-maps-activations.js','activationFunction activationBias rawActivationGrid afterActivationGrid activationName activationFormula activationBiasValue positiveRatio zeroRatio activationRange activationInterpretation',
    {activationFunction:'relu',activationBias:'-1.5'});
  const colors = html => [...html.matchAll(/data-value="([^"]+)"[^>]+style="background:([^"]+)"/g)].map(m => ({value:Number(m[1]),color:m[2]}));
  const before=colors(n.rawActivationGrid.innerHTML), after=colors(n.afterActivationGrid.innerHTML);
  assert.equal(before.length,49);
  assert.ok(before.some(x=>x.value>0));
  before.forEach((entry,i) => { if(entry.value>0) assert.equal(after[i].color,entry.color); });
});

test('the receptive-field recurrence distinguishes a previous stride from a new kernel', () => {
  const layers=[['input',1,1],['conv',3,1],['pool',2,2],['conv',3,1]];
  assert.deepEqual(math.receptiveFields(layers).map(s=>[s.receptiveField,s.jump]),[[1,1],[3,1],[4,2],[8,2]]);
  assert.equal(math.receptiveFields([['dilated',3,1,2]])[0].receptiveField,5);
});
