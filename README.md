# GNeural
A JavaScript library that allows you to easily create, train & run neural networks inside of Geometry Dash.

# Example
```js
import '@g-js-api/g.js';
import { MLPDense } from 'gneural';

// G.js export setup
await $.exportConfig({
    type: 'live_editor',
    options: {
        info: true,
        optimize: false
    }
});

// Create some training data for predicting the next number in a sequence
const trainingData = [
    { input: [1, 2], output: [3] },
    { input: [4, 5], output: [6] },
    { input: [7, 8], output: [9] },
    { input: [10, 11], output: [12] }
];

// Create a new neural network with two input neurons, two dense layers (4 neurons for first dense layer, 3 neurons for second) and one output neuron
// you can also provide a plain integer for second argument to create a singular dense layer
// or use the normal MLP class for singular hidden layer
const mlp = new MLPDense(2, [4, 3], 1); 
mlp.train(trainingData, 10000, 0.01); // Pre-trains the neural network before importing weights and biases into GD

// Test that your neural network works as expected
mlp.feedForward([1, 2]);
console.log(mlp.outputOutputs);

// Feed forward inside of Geometry Dash and store counters for displaying them 
// (you can also use the mlp.feedForwardFunction group to call without a pre-determined input, 
// but you will have to handle the counters in layers.input in-game if you want the player to be able to change inputs)

// you can also use counters instead of numbers as input for `mlp.predict()`
let layers = mlp.predict(1, 2);

// display the layers as counters
layers.input.forEach((x, i) => x.display(i * 70 + 45, 135))
layers.hidden.forEach((x, layer) => x.forEach((y, i) => y.display(i * 70 + 45, 105 - (layer * 30))));
layers.output.forEach((x, i) => x.display(i * 70 + 45, 105 - ((mlp.hasMultipleDenseLayers ? mlp.hiddenOutputCounters.length - 1 : 0) * 30) - 30));
```