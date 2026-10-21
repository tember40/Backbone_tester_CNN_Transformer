(function (root, factory) {
  const math = factory();
  if (typeof module === "object" && module.exports) module.exports = math;
  else root.SynthesisMath = math;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  function mix(farValue) {
    if (!Number.isFinite(farValue) || farValue < -2 || farValue > 2) throw new RangeError("farValue must be between -2 and 2.");
    const vectors = [[farValue, 0], [0, 1], [1, 1], [-1, 0], [0, -1]];
    const convWeights = [0, .25, .5, .25, 0];
    const scores = vectors.map(vector => (vector[0] + vector[1]) / Math.sqrt(2));
    const exponents = scores.map(score => Math.exp(score - Math.max(...scores)));
    const denominator = exponents.reduce((a, b) => a + b, 0);
    const attentionWeights = exponents.map(value => value / denominator);
    const weightedSum = weights => [0, 1].map(d => weights.reduce((total, w, i) => total + w * vectors[i][d], 0));
    return {vectors, scores, convWeights, attentionWeights, convOutput:weightedSum(convWeights), attentionOutput:weightedSum(attentionWeights)};
  }
  function summarizeMatrix(matrix) {
    if (!Array.isArray(matrix) || !matrix.length || matrix.some(row => !Array.isArray(row) || row.length !== matrix.length || row.some(n => !Number.isSafeInteger(n) || n < 0))) throw new TypeError("A nonnegative square count matrix is required.");
    const support = matrix.map(row => row.reduce((a, b) => a + b, 0));
    const total = support.reduce((a, b) => a + b, 0);
    const correct = matrix.reduce((sum, row, i) => sum + row[i], 0);
    const recalls = support.map((count, i) => count ? matrix[i][i] / count * 100 : null);
    const present = recalls.filter(value => value !== null);
    return {support, total, correct, recalls, accuracy:total ? correct / total * 100 : null,
      macroRecall:present.length ? present.reduce((a, b) => a + b, 0) / present.length : null};
  }
  const matrices = {
    balanced:[[8, 2, 0], [1, 7, 2], [0, 3, 7]],
    imbalanced:[[90, 0, 0], [5, 0, 0], [5, 0, 0]],
    missing:[[8, 2, 0], [1, 9, 0], [0, 0, 0]],
  };
  const curves = {
    gap:{train:[1.9, 1.4, 1., .7, .4, .2], val:[1.8, 1.45, 1.1, 1.15, 1.35, 1.55], accuracy:[35, 51, 65, 63, 61, 60],
      observation:"train loss는 계속 내려가지만 validation loss는 epoch3 이후 다시 올라갑니다.",
      hypothesis:"과적합 가능성뿐 아니라 train/eval 전처리와 데이터 분포 차이도 후보입니다.",
      action:"같은 eval 조건으로 train 일부를 평가하고 split·중복·전처리를 확인합니다. 정상이라면 증강이나 정규화를 한 가지씩 비교합니다."},
    flat:{train:[2.3, 2.25, 2.2, 2.2, 2.18, 2.15], val:[2.35, 2.3, 2.3, 2.28, 2.25, 2.25], accuracy:[11, 12, 13, 13, 14, 14],
      observation:"두 loss 모두 높은 수준에서 조금만 줄고 validation accuracy도 낮습니다.",
      hypothesis:"표현력 부족 외에 부적절한 학습률·라벨·gradient 문제나 부족한 학습 시간도 가능합니다.",
      action:"입력·라벨·gradient를 확인하고 소수 train 표본에서 loss를 낮출 수 있는지 먼저 시험합니다. 바로 더 큰 백본으로 바꾸지 않습니다."},
    unstable:{train:[1.8, 3.2, 1.1, 2.9, 1., 2.6], val:[1.9, 3.4, 1.2, 3., 1.1, 2.8], accuracy:[30, 18, 52, 20, 55, 24],
      observation:"두 loss와 validation accuracy가 epoch 사이에서 크게 오르내립니다.",
      hypothesis:"큰 학습률·배치 변동·이상 입력·수치 문제 등을 분리해 확인해야 합니다.",
      action:"입력 범위와 유한한 loss/gradient를 먼저 확인한 뒤, 같은 seed·데이터에서 학습률 하나를 낮춰 진동 변화를 비교합니다."},
  };
  function bestEpoch(accuracy) {
    if (!accuracy.length || accuracy.some(x => !Number.isFinite(x))) throw new TypeError("Finite validation accuracies are required.");
    return accuracy.reduce((best, value, index) => value > accuracy[best] ? index : best, 0) + 1;
  }
  return {mix, summarizeMatrix, matrices, curves, bestEpoch};
});
