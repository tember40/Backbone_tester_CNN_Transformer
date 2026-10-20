/* Pure arithmetic shared by the browser lesson and the numeric regression tests. */
((root) => {
  "use strict";

  function perceptronStep(weights, bias, sample, target, rate) {
    if (!weights.length || weights.length !== sample.length ||
        ![...weights, bias, ...sample, rate].every(Number.isFinite) ||
        ![0, 1].includes(target) || rate <= 0) {
      throw new RangeError("Use finite matching vectors, a binary target, and a positive rate.");
    }
    const dot = (w, x) => w.reduce((sum, value, index) => sum + value * x[index], 0);
    const scoreBefore = dot(weights, sample) + bias;
    const prediction = Number(scoreBefore >= 0);
    const error = target - prediction;
    const deltaWeights = sample.map((value) => rate * error * value);
    const deltaBias = rate * error;
    const weightsAfter = weights.map((value, index) => value + deltaWeights[index]);
    const biasAfter = bias + deltaBias;
    const scoreAfter = dot(weightsAfter, sample) + biasAfter;
    return {
      weightsBefore: [...weights], biasBefore: bias, sample: [...sample], target, rate,
      scoreBefore, prediction, error, deltaWeights, deltaBias, weightsAfter, biasAfter,
      scoreAfter, predictionAfter: Number(scoreAfter >= 0),
    };
  }

  const api = Object.freeze({ perceptronStep });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BackboneFoundations = api;
})(globalThis);
