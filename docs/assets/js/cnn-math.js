/* Small, deterministic calculations used by the CNN lessons and numeric tests. */
((root) => {
  "use strict";
  const positiveInteger = (value) => Number.isInteger(value) && value > 0;
  function outputSize(input, kernel, stride = 1, padding = 0, dilation = 1) {
    if (![input, kernel, stride, dilation].every(positiveInteger) ||
        !Number.isInteger(padding) || padding < 0) throw new RangeError("Invalid layer geometry.");
    return Math.max(0, Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1);
  }
  function matrixShape(matrix) {
    const width = matrix[0]?.length;
    if (!matrix.length || !width || matrix.some(row => row.length !== width || !row.every(Number.isFinite))) {
      throw new RangeError("Use a nonempty rectangular finite matrix.");
    }
    return [matrix.length, width];
  }
  function correlate2D(image, kernel, { stride = 1, padding = 0 } = {}) {
    const [h, w] = matrixShape(image), [kh, kw] = matrixShape(kernel);
    const oh = outputSize(h, kh, stride, padding), ow = outputSize(w, kw, stride, padding);
    if (!oh || !ow) throw new RangeError("Kernel is larger than padded input.");
    return Array.from({length: oh}, (_, y) => Array.from({length: ow}, (_, x) =>
      kernel.reduce((sum, row, ky) => sum + row.reduce((part, weight, kx) => {
        const iy = y * stride + ky - padding, ix = x * stride + kx - padding;
        return part + (iy >= 0 && iy < h && ix >= 0 && ix < w ? image[iy][ix] : 0) * weight;
      }, 0), 0)
    ));
  }
  function pool2D(matrix, kernel = 2, stride = 2, mode = "max") {
    const [h, w] = matrixShape(matrix);
    if (!["max", "average"].includes(mode)) throw new RangeError("Unknown pooling mode.");
    const oh = outputSize(h, kernel, stride), ow = outputSize(w, kernel, stride);
    if (!oh || !ow) throw new RangeError("Pooling window is larger than input.");
    return Array.from({length: oh}, (_, y) => Array.from({length: ow}, (_, x) => {
      const values = matrix.slice(y * stride, y * stride + kernel)
        .flatMap(row => row.slice(x * stride, x * stride + kernel));
      return mode === "max" ? Math.max(...values) : values.reduce((a, b) => a + b, 0) / values.length;
    }));
  }
  function poolingGeometry(input, kernel, stride) {
    const output = outputSize(input, kernel, stride);
    const axis = Array(input).fill(false);
    for (let start = 0; start < output; start++) {
      for (let offset = 0; offset < kernel; offset++) axis[start * stride + offset] = true;
    }
    const coveredAxis = axis.filter(Boolean).length;
    return {output, axis, uncovered: input * input - coveredAxis * coveredAxis,
      trailing: output ? input - ((output - 1) * stride + kernel) : input};
  }
  function receptiveFields(layers) {
    let receptiveField = 1, jump = 1;
    return layers.map(([name, kernel, stride, dilation = 1]) => {
      if (![kernel, stride, dilation].every(positiveInteger)) throw new RangeError("Invalid receptive-field stage.");
      const previousJump = jump;
      receptiveField += dilation * (kernel - 1) * jump;
      jump *= stride;
      return {name, kernel, stride, dilation, receptiveField, jump, previousJump};
    });
  }
  const api = Object.freeze({outputSize, correlate2D, pool2D, poolingGeometry, receptiveFields});
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BackboneCNNMath = api;
})(globalThis);
