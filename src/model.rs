use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

// Instantiate the model for training
#[derive(Debug, Config)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, widht] = images.dims();

        let x = images.reshape([batch_size, 1, height, widht]);

        let x = self.conv1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}

// This Rust code defines a simple convolutional neural network (CNN) model using the burn deep learning framework. Here's a breakdown of what it does:

// 1. Struct Definitions
// Model<B: Backend>: Defines the CNN model, which consists of:

// Two convolutional layers (conv1, conv2) for feature extraction.
// One adaptive average pooling layer (pool) to reduce spatial dimensions.
// One dropout layer (dropout) to prevent overfitting.
// Two fully connected (linear) layers (linear1, linear2) for classification.
// A ReLU activation function (activation).
// ModelConfig: A configuration structure that holds:

// num_classes: Number of output classes for classification.
// hidden_size: Number of neurons in the first fully connected layer.
// dropout: Dropout rate (default 0.5).
// 2. Initialization (init method)
// Creates and initializes the layers with the given device (CPU/GPU).
// The network structure:
// conv1: 1 input channel → 8 output channels, kernel size (3,3).
// conv2: 8 input channels → 16 output channels, kernel size (3,3).
// pool: Adaptive average pooling to (8,8).
// linear1: Fully connected layer from 16 × 8 × 8 → hidden_size.
// linear2: Fully connected layer from hidden_size → num_classes.
// dropout: Applied before fully connected layers.
// Usage
// ModelConfig is used to initialize a Model with a specific number of classes and hidden layer size.
// This model can be trained for image classification tasks.
