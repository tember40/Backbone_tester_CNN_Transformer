const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Run the real controllers. This adapter only supplies DOM events and display nodes.
function lesson(script, ids, values = {}) {
  const nodes = Object.fromEntries(ids.split(' ').map(id => {
    const handlers = {};
    return [id, {value:values[id] ?? '', textContent:'', innerHTML:'', disabled:false,
      style:{setProperty() {}}, querySelectorAll:() => [],
      addEventListener(type, fn) {handlers[type] = fn;},
      emit(type) {handlers[type]?.();}, click() {handlers.click?.();},
    }];
  }));
  const sandbox = {document:{querySelector:s => nodes[s.slice(1)] ?? null, querySelectorAll:() => []},
    IntersectionObserver:class {observe() {}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../docs/assets/js', script), 'utf8'), sandbox);
  return nodes;
}

const dropoutIds = 'dropoutProbability dropoutUnits dropoutMode resampleDropout dropoutProbabilityValue dropoutActive dropoutScale dropoutMessage';
const dropoutValues = nodes => [...nodes.dropoutUnits.innerHTML.matchAll(/data-input="([^"]+)" data-output="([^"]+)"/g)]
  .map(match => [Number(match[1]), Number(match[2])]);

test('Dropout shows the actual masked and rescaled elements and eval bypasses the mask', () => {
  const n = lesson('alexnet.js', dropoutIds, {dropoutProbability:'0.5', dropoutMode:'train'});
  const first = dropoutValues(n);
  assert.equal(first.length, 24);
  assert.ok(first.some(([,output]) => output === 0));
  assert.ok(first.some(([input,output]) => output === input * 2));
  first.forEach(([input,output]) => assert.ok(output === 0 || output === input * 2));
  assert.match(n.dropoutMessage.textContent, /정확히 50%/);
  n.resampleDropout.click();
  assert.notDeepEqual(dropoutValues(n), first);
  n.dropoutMode.value = 'eval'; n.dropoutMode.emit('change');
  dropoutValues(n).forEach(([input,output]) => assert.equal(input, output));
  assert.equal(n.dropoutActive.textContent, '24 / 24');
  assert.equal(n.dropoutScale.textContent, '1.00배');
  assert.equal(n.resampleDropout.disabled, true);
  n.dropoutProbability.value = '0.8'; n.dropoutProbability.emit('input');
  dropoutValues(n).forEach(([input,output]) => assert.equal(input, output));
  n.dropoutMode.value = 'train'; n.dropoutMode.emit('change');
  assert.equal(n.resampleDropout.disabled, false);
  assert.equal(n.dropoutScale.textContent, '5.00배');
});

test('Dropout p=0 has no removals and reloading starts from the same illustrative mask', () => {
  const a = lesson('alexnet.js', dropoutIds, {dropoutProbability:'0.5', dropoutMode:'train'});
  const b = lesson('alexnet.js', dropoutIds, {dropoutProbability:'0.5', dropoutMode:'train'});
  assert.deepEqual(dropoutValues(a), dropoutValues(b));
  a.dropoutProbability.value = '0'; a.dropoutProbability.emit('input');
  dropoutValues(a).forEach(([input,output]) => assert.equal(input, output));
  assert.equal(a.dropoutActive.textContent, '24 / 24');
  assert.equal(a.dropoutScale.textContent, '1.00배');
});

test('AlexNet classifier combines all spatial locations rather than keeping one-cell RF', () => {
  const n = lesson('alexnet.js', 'alexStage alexStageList alexStageValue alexLayerName alexLayerShape alexLayerParams alexLayerRF alexLayerNote alexFeatureStack', {alexStage:'8'});
  assert.equal(n.alexLayerRF.textContent, '42 × 42');
  n.alexStage.value = '9'; n.alexStage.emit('input');
  assert.equal(n.alexLayerShape.textContent, '[10]');
  assert.equal(n.alexLayerRF.textContent, '66 × 66');
  assert.match(n.alexLayerNote.textContent, /4×4의 모든 위치/);
});

test('VGG parameter counts and memory remain distinct as batch and depth change', () => {
  const n = lesson('vgg.js', 'smallKernelLayers smallKernelValue kernelReceptive kernelStackParameters kernelWideParameters kernelDifference kernelStack kernelFootnote memoryBatch memoryVariant memoryBatchValue memoryWeights memoryFeature memoryNote',
    {smallKernelLayers:'2', memoryBatch:'32', memoryVariant:'vgg16'});
  assert.equal(n.kernelReceptive.textContent, '5 × 5');
  assert.equal(n.kernelStackParameters.textContent, '73,856');
  assert.equal(n.kernelWideParameters.textContent, '102,464');
  assert.equal(n.memoryWeights.textContent, '128.3 MiB');
  assert.equal(n.memoryFeature.textContent, '8.0 MiB');
  n.memoryBatch.value = '64'; n.memoryBatch.emit('input');
  assert.equal(n.memoryFeature.textContent, '16.0 MiB');
  assert.equal(n.memoryWeights.textContent, '128.3 MiB');
  n.smallKernelLayers.value = '1'; n.smallKernelLayers.emit('input');
  assert.equal(n.kernelStackParameters.textContent, n.kernelWideParameters.textContent);
  n.memoryVariant.value = 'vgg11'; n.memoryVariant.emit('change');
  assert.equal(n.memoryWeights.textContent, '107.4 MiB');
  assert.equal(n.memoryFeature.textContent, '16.0 MiB');
});

test('VGG block controller traces all five stages for every listed depth', () => {
  const n = lesson('vgg.js', 'vggVariant vggBlock vggVariantName vggDepth vggParameterTotal vggBlockValue vggBlockName vggBlockLayers vggBlockShape vggBlockBefore vggBlockAfter vggBlockRF vggBlockParameters vggBlockList', {vggVariant:'vgg16', vggBlock:'0'});
  assert.equal(n.vggBlockShape.textContent, '[64, 16, 16]');
  assert.equal(n.vggBlockRF.textContent, '6×6');
  const cases = {vgg11:['28,144,010',150],vgg13:['28,328,522',156],vgg16:['33,638,218',212],vgg19:['38,947,914',268]};
  for (const [variant,[parameters,rf]] of Object.entries(cases)) {
    n.vggVariant.value = variant; n.vggVariant.emit('change');
    assert.equal(n.vggBlockValue.textContent, '1 / 5');
    assert.equal(n.vggParameterTotal.textContent, parameters);
    n.vggBlock.value = '4'; n.vggBlock.emit('input');
    assert.equal(n.vggBlockShape.textContent, '[512, 1, 1]');
    assert.equal(n.vggBlockRF.textContent, `${rf}×${rf}`);
  }
});

test('ResNet stage explorer projects when channels change even without spatial downsampling', () => {
  const n = lesson('resnet.js', 'resnetVariant resnetStageList resnetBlockPattern resnetBlockType resnetParameters resnetStageName resnetStageDescription resnetStageInput resnetStageOutput resnetStageBlocks resnetStageShortcut', {resnetVariant:'resnet18'});
  assert.equal(n.resnetStageName.textContent, 'layer1');
  assert.equal(n.resnetStageShortcut.textContent, 'identity (그대로 전달)');
  n.resnetVariant.value = 'resnet50'; n.resnetVariant.emit('change');
  assert.equal(n.resnetStageInput.textContent, '[64, 8, 8]');
  assert.equal(n.resnetStageOutput.textContent, '[256, 8, 8]');
  assert.equal(n.resnetStageShortcut.textContent, '1×1 Conv + BatchNorm');
  assert.equal(n.resnetParameters.textContent, '23,528,522');
});

test('ResNet scalar demonstration keeps the final ReLU and its identity limits', () => {
  const n = lesson('resnet.js', 'residualInput residualChange residualInputValue residualChangeValue residualDiagramInput residualDiagramChange residualDiagramSkip residualSum residualOutput plainOutput residualNote', {residualInput:'1', residualChange:'0'});
  assert.equal(n.residualOutput.textContent, '1.0');
  assert.equal(n.plainOutput.textContent, '0.0');
  n.residualChange.value = '-1.5'; n.residualChange.emit('input');
  assert.equal(n.residualSum.textContent, '-0.5');
  assert.equal(n.residualOutput.textContent, '0.0');
  assert.match(n.residualNote.textContent, /무조건 원본 값을 보존하는 스위치가 아닙니다/);
});
