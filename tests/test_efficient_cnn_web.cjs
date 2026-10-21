const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Exercise the page controllers; the adapter supplies only nodes and input events.
function lesson(script, ids, values) {
  const nodes = Object.fromEntries(ids.split(' ').map(id => {
    const handlers = {};
    return [id, {id, value:values[id] ?? '', textContent:'', innerHTML:'', disabled:false, style:{},
      addEventListener(type, fn) {handlers[type] = fn;},
      emit(type, event) {handlers[type]?.(event);},
    }];
  }));
  const selectAll = selector => selector.startsWith('#')
    ? selector.split(',').map(part => nodes[part.trim().slice(1)]).filter(Boolean) : [];
  const sandbox = {document:{querySelector:s => nodes[s.slice(1)] ?? null, querySelectorAll:selectAll},
    IntersectionObserver:class {observe() {}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../docs/assets/js',script),'utf8'), sandbox);
  return nodes;
}

test('SE fixed MLP connects means, hidden value, gates and output; manual gates survive mode changes', () => {
  const n = lesson('senet.js','seChannel sePixelGrid seSqueezeSummary seSqueezeEquation seGate1 seGate2 seGate3 seGate1Value seGate2Value seGate3Value seCurrentGate seScaleBefore seScaleAfter seScaleExplanation seGateMode seMLPMeans seMLPHidden seMLPLogits seMLPGates seMLPMessage',
    {seChannel:'0', seGateMode:'manual'});
  assert.equal(n.seMLPMeans.textContent, '[4.0000, 3.0000, 3.0000]');
  assert.equal(n.seMLPHidden.textContent, '1.2500 → ReLU → 1.2500');
  assert.equal(n.seMLPGates.textContent, '[0.7773, 0.6514, 0.2227]');
  assert.equal(n.seGate1Value.textContent, '0.25');
  n.seGate1.value = '0.4'; n.seGate1.emit('input');
  n.seGateMode.value = 'mlp'; n.seGateMode.emit('change');
  assert.equal(n.seGate1.disabled, true);
  assert.equal(n.seGate1Value.textContent, '0.7773');
  assert.match(n.seScaleAfter.innerHTML, /<span>0.78<\/span>/);
  const field = {value:'-9',dataset:{position:'0'},closest() {return this;}};
  n.sePixelGrid.emit('input', {target:field});
  assert.equal(n.seMLPHidden.textContent, '0.0000 → ReLU → 0.0000');
  assert.equal(n.seMLPGates.textContent, '[0.5000, 0.5000, 0.5000]');
  assert.match(n.seScaleAfter.innerHTML, /<span>-4.50<\/span>/);
  n.seGateMode.value = 'manual'; n.seGateMode.emit('change');
  assert.equal(n.seGate1.disabled, false);
  assert.equal(n.seGate1Value.textContent, '0.40');
  field.value='-100'; n.sePixelGrid.emit('input',{target:field});
  assert.match(n.seSqueezeEquation.textContent, /\(-9 \+ 3/);
  const gridBeforeCommit = n.sePixelGrid.innerHTML;
  n.sePixelGrid.emit('change',{target:field});
  assert.equal(field.value,'-9');
  assert.equal(n.sePixelGrid.innerHTML,gridBeforeCommit); // Preserve the focused input node.
});

test('SE reduction displays actual bias-free matrix counts', () => {
  const n = lesson('senet.js','seChannels seReduction seDimInput seDimHidden seDimOutput seParameterExplanation', {seChannels:'64',seReduction:'16'});
  assert.equal(n.seDimHidden.textContent, '4');
  assert.match(n.seParameterExplanation.textContent,/512개/);
  n.seChannels.value = '512'; n.seReduction.value = '32'; n.seReduction.emit('change');
  assert.equal(n.seDimHidden.textContent, '16');
  assert.match(n.seParameterExplanation.textContent,/16,384개/);
});

test('MobileNet complete block shows expansion cost and stride-specific shapes', () => {
  const n = lesson('mobilenetv2.js','mobileBlockInput mobileBlockOutput mobileExpand mobileStride mobileFlowInput mobileFlowHidden mobileFlowDepthwise mobileFlowOutput mobileFlowExpandLabel mobileExpandNote mobileSkipNote mobileBlockShapes mobileBlockWeights',
    {mobileBlockInput:'24',mobileBlockOutput:'24',mobileExpand:'6',mobileStride:'1'});
  assert.match(n.mobileBlockWeights.textContent,/3,456 \+ depthwise 1,296 \+ 투영 3,456 = 8,208개/);
  assert.match(n.mobileBlockShapes.textContent,/\[N,144,8,8\] → \[N,24,8,8\]/);
  assert.match(n.mobileSkipNote.textContent,/입력을 더합니다/);
  n.mobileStride.value='2'; n.mobileStride.emit('change');
  assert.match(n.mobileBlockShapes.textContent,/\[N,144,4,4\] → \[N,24,4,4\]/);
  assert.match(n.mobileSkipNote.textContent,/더하지 않습니다/);
  n.mobileExpand.value='1'; n.mobileExpand.emit('change');
  assert.equal(n.mobileFlowExpandLabel.textContent,'확장 생략');
  assert.match(n.mobileBlockWeights.textContent,/확장 0 \+ depthwise 216 \+ 투영 576 = 792개/);
  n.mobileStride.value='1'; n.mobileBlockOutput.value='16'; n.mobileStride.emit('change');
  assert.match(n.mobileSkipNote.textContent,/더하지 않습니다/);
});

test('MobileNet width rounding and stage selection track the real settings', () => {
  const n = lesson('mobilenetv2.js','mobileWidth mobileStageList mobileParameters mobileLastChannels mobileStageName mobileStagePattern mobileStageOutput mobileStageStride mobileStageRepeats mobileStageExpand', {mobileWidth:'0.75'});
  assert.equal(n.mobileParameters.textContent,'1,368,234');
  const stageButton = {dataset:{index:'2'},closest() {return this;}};
  n.mobileStageList.emit('click',{target:stageButton});
  assert.equal(n.mobileStageOutput.textContent,'[24, 8, 8]');
  stageButton.dataset.index='8'; n.mobileStageList.emit('click',{target:stageButton});
  assert.equal(n.mobileStageOutput.textContent,'[1280, 1, 1]');
});

test('EfficientNet continuous scaling has a known numerical result and is not preset selection', () => {
  const n = lesson('efficientnet.js','efficientPhi efficientPhiValue efficientDepth efficientWidth efficientResolution efficientDepthBar efficientWidthBar efficientResolutionBar efficientCost', {efficientPhi:'2'});
  assert.equal(n.efficientDepth.textContent,'1.44×');
  assert.equal(n.efficientWidth.textContent,'1.21×');
  assert.equal(n.efficientResolution.textContent,'1.32×');
  assert.match(n.efficientCost.textContent,/3.69배.*실측 FLOPs가 아닙니다/);
  n.efficientPhi.value='0'; n.efficientPhi.emit('input');
  assert.match(n.efficientCost.textContent,/1.00배/);
});

test('EfficientNet integer stage repeats, stride and channels match the shown B variants', () => {
  const n = lesson('efficientnet.js','efficientPreset efficientStageList efficientMultipliers efficientBlocks efficientParameters efficientStageName efficientStagePattern efficientStageOutput efficientStageRepeats efficientStageStride efficientStageKernel', {efficientPreset:'b1'});
  assert.equal(n.efficientBlocks.textContent,'23개');
  assert.equal(n.efficientStageRepeats.textContent,'2개');
  assert.equal(n.efficientParameters.textContent,'6,574,186');
  n.efficientPreset.value='b2'; n.efficientPreset.emit('change');
  assert.equal(n.efficientStageOutput.textContent,'[16, 16, 16]');
  const button={dataset:{index:'3'},closest() {return this;}};
  n.efficientStageList.emit('click',{target:button});
  assert.equal(n.efficientStageOutput.textContent,'[48, 4, 4]');
  assert.equal(n.efficientStageStride.textContent,'2');
  assert.equal(n.efficientStageKernel.textContent,'5×5');
  button.dataset.index='8'; n.efficientStageList.emit('click',{target:button});
  assert.equal(n.efficientStageOutput.textContent,'[1408, 1, 1]');
});
