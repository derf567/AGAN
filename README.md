🌧️ Attentive GAN DerainNet

A TensorFlow implementation for removing rain streaks and raindrops from single images using an Attentive Generative Adversarial Network.

📘 Overview

Attentive GAN DerainNet is a deep-learning toolkit designed to remove rain, rain streaks, and raindrops from images.
It is based on the CVPR 2018 paper:

Attentive Generative Adversarial Network for Raindrop Removal from a Single Image

This framework integrates attention mechanisms, ConvLSTM recurrence, and adversarial learning to produce clean and visually consistent results.

✨ Features

✔️ Removes rain streaks and raindrop artifacts

✔️ Attentive recurrent ConvLSTM generator

✔️ Contextual autoencoder for fine-detail restoration

✔️ Adversarial training (GAN)

✔️ Multi-loss optimization (perceptual, multi-scale, GAN loss)

✔️ Ready-to-use inference tools

✔️ TFRecords dataset preparation

✔️ Export model to SavedModel or TensorFlow.js

✔️ TensorBoard support

📦 Installation
Requirements

Python 3.5+

TensorFlow 1.15

CUDA + cuDNN (optional, for GPU)

All dependencies listed in requirements.txt

Steps

Clone the repository:

git clone https://github.com/derf567/attentive-gan-derainnet
cd attentive-gan-derainnet


Install dependencies:

pip install -r requirements.txt


(Optional) Install CUDA & cuDNN for GPU acceleration.

🧪 Inference (Testing)

Run deraining on a single image:

python tools/test_model.py \
--weights_path ./weights/derain_gan/derain_gan.ckpt-xxxxxx \
--image_path PATH_TO_IMAGE


Replace PATH_TO_IMAGE with your test image.

Ensure your checkpoint path is correct.

Output includes:

Clean derained image

Attention map (highlighting rain regions)

🏋️ Training
Dataset Structure
dataset_root/
    rain_image/   # Rainy input images
    clean_image/  # Ground-truth clean images

Convert dataset to TFRecords
python data_provider/data_feed_pipline.py \
--dataset_dir DATASET_ROOT \
--tfrecords_dir TFRECORDS_SAVE_DIR

Start Training
python tools/train_model.py --dataset_dir DATASET_ROOT

Recommended Settings (global_configuration/config.py)

Initial LR: 0.002

Batch size: 1 (no batch norm required)

Training iterations: 100,000+

🛰️ Exporting the Model
Export to TensorFlow SavedModel:
python tools/export_tf_saved_model.py \
--export_dir SAVE_DIR \
--ckpt_path CKPT_PATH

Convert to TensorFlow.js:
bash tools/convert_tfjs_model.sh

📊 Monitoring with TensorBoard
tensorboard --logdir=LOG_DIR


Tracks:

Training losses

Sample outputs

Attention visualization

🧠 Technical Architecture
Generator

Attentive recurrent network

ConvLSTM for temporal/spatial attention

Contextual autoencoder

Discriminator

CNN-based discriminator

Attention Mechanism

Identifies and focuses on rain/raindrop regions for precise restoration.

Loss Functions

Multi-scale reconstruction loss

Perceptual loss

Adversarial loss

🗂️ Main Scripts
Script	Description
tools/train_model.py	Training entry point
tools/test_model.py	Run inference on images
data_provider/data_feed_pipline.py	Converts dataset → TFRecords
tools/export_tf_saved_model.py	Export as TensorFlow SavedModel
tools/convert_tfjs_model.sh	Convert to TFJS format
global_configuration/config.py	Central config: LR, epochs, paths
📚 Reference

Attentive Generative Adversarial Network for Raindrop Removal from a Single Image
CVPR 2018
