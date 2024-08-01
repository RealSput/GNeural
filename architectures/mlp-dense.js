import MLP from './mlp.js';
let gapnum = 0;
let gap = () => {
    let currCtx = Context.findByName(Context.current);
    let lastObj = currCtx.objects[currCtx.objects.length - 1];
    if (!lastObj?.X) {
        lastObj.X = gapnum;
        return;
    }
    lastObj.X += gapnum;
    gapnum += 1;
}
class MLPDense extends MLP {
    constructor(inputNeurons, hiddenNeurons, outputNeurons) {
        super(inputNeurons, Array.isArray(hiddenNeurons) ? hiddenNeurons[0] : hiddenNeurons, outputNeurons);
        this.hasMultipleDenseLayers = Array.isArray(hiddenNeurons);
        this.hiddenLayersConfig = Array.isArray(hiddenNeurons) ? hiddenNeurons : [hiddenNeurons];
        this.weightsHidden = [this.weightsInputHidden];
        this.biasHidden = [this.biasHidden];
        
        for (let i = 1; i < this.hiddenLayersConfig.length; i++) {
            this.weightsHidden.push(this.initializeWeights(this.hiddenLayersConfig[i - 1], this.hiddenLayersConfig[i]));
            this.biasHidden.push(new Array(this.hiddenLayersConfig[i]).fill(1));
        }
        
        this.weightsHiddenOutput = this.initializeWeights(this.hiddenLayersConfig[this.hiddenLayersConfig.length - 1], outputNeurons);
        this.hiddenOutputs = this.hiddenLayersConfig.map(neurons => new Array(neurons).fill(0));
        this.hiddenOutputCounters = this.hiddenOutputs.map(x => x.map(_ => float_counter()));
        this.hiddenGradients = this.hiddenLayersConfig.map(neurons => new Array(neurons).fill(0));
    }

    relu(x) {
        return Math.max(0, x);
    }

    passThroughHiddenLayers() {
        for (let layer = 0; layer < this.hiddenLayersConfig.length; layer++) {
            const input = layer === 0 ? this.inputs : this.hiddenOutputs[layer - 1];
            const weights = this.weightsHidden[layer];
            const bias = this.biasHidden[layer];

            for (let i = 0; i < this.hiddenLayersConfig[layer]; i++) {
                let sum = 0;
                for (let j = 0; j < input.length; j++) {
                    sum += input[j] * weights[j][i];
                }
                sum += bias[i];
                this.hiddenOutputs[layer][i] = this.relu(sum);
            }
        }
    }

    GD_passThroughHiddenLayers() {
        for (let layer = 0; layer < this.hiddenLayersConfig.length; layer++) {
            const input = layer === 0 ? this.inputNeuronCounters : this.hiddenOutputCounters[layer - 1];
            const weights = this.weightsHidden[layer];
            const bias = this.biasHidden[layer];
            for (let i = 0; i < this.hiddenLayersConfig[layer]; i++) {
                for (let j = 0; j < input.length; j++) {
                    $.add(item_edit(input[j].item, 0, this.hiddenOutputCounters[layer][i].item, TIMER, NONE, TIMER, ADD, MUL, NONE, weights[j][i]));
                }
                this.hiddenOutputCounters[layer][i].add(bias[i]);
                compare(this.hiddenOutputCounters[layer][i], LESS, 0, trigger_function(() => this.hiddenOutputCounters[layer][i].set(0)));
                gap();
            }
        }
    }

    passThroughOutputLayer() {
        const lastHiddenLayer = this.hiddenOutputs[this.hiddenOutputs.length - 1];
        for (let i = 0; i < this.outputNeurons; i++) {
            let sum = 0;
            for (let j = 0; j < lastHiddenLayer.length; j++) {
                sum += lastHiddenLayer[j] * this.weightsHiddenOutput[j][i];
            }
            sum += this.biasOutput[i];
            this.outputOutputs[i] = this.relu(sum);
        }
    }

    GD_passThroughOutputLayer() {
        const lastHiddenLayer = this.hiddenOutputCounters[this.hiddenOutputs.length - 1];
        for (let i = 0; i < this.outputNeurons; i++) {
            for (let j = 0; j < lastHiddenLayer.length; j++) {
                $.add(item_edit(lastHiddenLayer[j].item, 0, this.outputOutputCounters[i].item, TIMER, TIMER, TIMER, ADD, MUL, NONE, this.weightsHiddenOutput[j][i]));
            }
            this.outputOutputCounters[i].add(this.biasOutput[i]);
            gap();
            compare(this.outputOutputCounters[i], LESS, 0, trigger_function(() => this.outputOutputCounters[i].set(0)));
            gap();
        }
    }

    passErrorToHiddenLayers() {
        for (let layer = this.hiddenLayersConfig.length - 1; layer >= 0; layer--) {
            const nextLayer = layer === this.hiddenLayersConfig.length - 1 ? this.outputGradients : this.hiddenGradients[layer + 1];
            const weights = layer === this.hiddenLayersConfig.length - 1 ? this.weightsHiddenOutput : this.weightsHidden[layer + 1];

            for (let i = 0; i < this.hiddenLayersConfig[layer]; i++) {
                let errorGradient = 0;
                for (let j = 0; j < nextLayer.length; j++) {
                    errorGradient += nextLayer[j] * weights[i][j];
                }
                this.hiddenGradients[layer][i] = errorGradient * (this.hiddenOutputs[layer][i] > 0 ? 1 : 0);
            }
        }
    }

    passErrorToOutputLayer(target) {
        for (let i = 0; i < this.outputNeurons; i++) {
            const error = target[i] - this.outputOutputs[i];
            this.outputGradients[i] = error * (this.outputOutputs[i] > 0 ? 1 : 0);
        }
    }

    feedForward(inputData) {
        this.setInputData(inputData);
        this.passThroughHiddenLayers();
        this.passThroughOutputLayer();
    }

    GD_feedForward(inputData) {
        if (!this.feedForwardFunction) this.feedForwardFunction = trigger_function(() => {
            this.GD_passThroughHiddenLayers();
            gap();
            this.GD_passThroughOutputLayer();
        });
        this.GD_setInputData(inputData);
        this.feedForwardFunction.call();
        gap();
    }

    train(trainingData, iterations, learningRate = 0.001, regularizationRate = 0.01, noiseLevel = 0.01) {
        for (let i = 0; i < iterations; i++) {
            for (const data of trainingData) {
                const noisyInput = data.input.map(x => x + (Math.random() - 0.5) * noiseLevel);
                this.feedForward(noisyInput);
                const error = this.calculateError(data.output);
                this.passErrorToOutputLayer(data.output);
                this.passErrorToHiddenLayers();
                this.updateWeights(learningRate, regularizationRate);
            }
        }
    }

    updateWeights(learningRate, regularizationRate) {
        const clipValue = 1.0;

        for (let layer = 0; layer < this.hiddenLayersConfig.length; layer++) {
            const input = layer === 0 ? this.inputs : this.hiddenOutputs[layer - 1];
            for (let i = 0; i < input.length; i++) {
                for (let j = 0; j < this.hiddenLayersConfig[layer]; j++) {
                    let gradient = this.hiddenGradients[layer][j];
                    if (gradient > clipValue) gradient = clipValue;
                    if (gradient < -clipValue) gradient = -clipValue;
                    const regularization = regularizationRate * this.weightsHidden[layer][i][j];
                    this.weightsHidden[layer][i][j] += learningRate * (input[i] * gradient - regularization);
                }
            }
        }

        const lastHiddenLayer = this.hiddenOutputs[this.hiddenOutputs.length - 1];
        for (let i = 0; i < lastHiddenLayer.length; i++) {
            for (let j = 0; j < this.outputNeurons; j++) {
                let gradient = this.outputGradients[j];
                if (gradient > clipValue) gradient = clipValue;
                if (gradient < -clipValue) gradient = -clipValue;
                const regularization = regularizationRate * this.weightsHiddenOutput[i][j];
                this.weightsHiddenOutput[i][j] += learningRate * (lastHiddenLayer[i] * gradient - regularization);
            }
        }
    }
}

export default MLPDense;
