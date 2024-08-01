import '@g-js-api/g.js';
import MLPDense from './mlp-dense.js';

await $.exportConfig({
    type: 'live_editor',
    options: {
        info: true,
        optimize: false
    }
});

// Create some training data (XOR problem)
const trainingData = [
    { input: [1, 2], output: [3] },
    { input: [4, 5], output: [6] },
    { input: [7, 8], output: [9] },
    { input: [8, 9], output: [11] }
];

const mlp = new MLPDense(2, [4, 3], 1);
mlp.train(trainingData, 10000, 0.01); // train before importing parameters into GD
mlp.feedForward([1, 2]);
console.log(mlp.outputOutputs);
wait(1);
let layers = mlp.predict([1, 2]);
layers.input.forEach((x, i) => x.display(i * 70 + 45, 135))
layers.hidden.forEach((x, layer) => x.forEach((y, i) => y.display(i * 70 + 45, 105 - (layer * 30))))
layers.output.forEach((x, i) => x.display(i * 70 + 45, 105 - ((mlp.hiddenOutputCounters.length - 1) * 30) - 30));
