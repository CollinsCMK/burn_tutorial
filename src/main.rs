use burn::{backend::{Autodiff, Wgpu}, data::dataset::Dataset, optim::AdamConfig};
use std::env;

use model::ModelConfig;
use training::TrainingConfig;

mod data;
mod model;
mod training;
mod inference;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";

    // Get command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --release train | infer");
        return;
    }

    match args[1].as_str() {
        "train" => {
            println!("Starting Training...");
            crate::training::train::<MyAutodiffBackend>(
                artifact_dir,
                TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
                device.clone(),
            );
            println!("Training Complete!");
        }
        "infer" => {
            println!("Running Inference...");
            let mnist_sample = burn::data::dataset::vision::MnistDataset::test()
                .get(12)
                .unwrap();

            crate::inference::infer::<MyBackend>(
                artifact_dir,
                device,
                mnist_sample,
            );
        }
        _ => {
            println!("Invalid command! Use 'train' or 'infer'.");
        }
    }
}
