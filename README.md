Attentive GAN DerainNet

A Python-based deep learning toolkit for removing rain and raindrop artifacts from single images using an Attentive Generative Adversarial Network (GAN).
The framework is based on the CVPR 2018 paper:
“Attentive Generative Adversarial Network for Raindrop Removal from a Single Image.”

✨ Features

Removes rain streaks and raindrops from single images

Generator uses attentive recurrent ConvLSTM + contextual autoencoder

Adversarial training with multi-loss optimization

Includes tools for training, inference, TFRecords dataset preparation, and export to TensorFlow SavedModel / TFJS

TensorBoard monitoring support

📦 Installation
Requirements

Python 3.5+

TensorFlow 1.15 (recommended)

CUDA + cuDNN (optional, for GPU acceleration)

Additional dependencies in requirements.txt

Steps

Clone the repository:

git clone https://github.com/derf567/attentive-gan-derainnet
cd attentive-gan-derainnet


Install dependencies:

pip install -r requirements.txt


(Optional) Install and configure CUDA and cuDNN for GPU support.

🧪 Inference (Testing)

Run deraining on a single image:

python tools/test_model.py \
    --weights_path ./weights/derain_gan/derain_gan.ckpt-xxxxxx \
    --image_path PATH_TO_IMAGE


Replace PATH_TO_IMAGE with your input image and update the checkpoint path accordingly.

🏋️ Training
1. Prepare Dataset

Organize training images into:

dataset_root/
    rain_image/    # Rainy input images
    clean_image/   # Ground-truth clean images

2. Convert to TFRecords
python data_provider/data_feed_pipline.py \
    --dataset_dir DATASET_ROOT \
    --tfrecords_dir TFRECORDS_SAVE_DIR

3. Train the Model
python tools/train_model.py --dataset_dir DATASET_ROOT


Configure learning rate, epochs, batch size, and paths in:

global_configuration/config.py


Recommended training settings:

Initial learning rate: 0.002

Batch size: 1 (no batch normalization needed)

Training iterations: 100,000+

🛰️ Exporting Models
Export to TensorFlow SavedModel
python tools/export_tf_saved_model.py \
    --export_dir SAVE_DIR \
    --ckpt_path CKPT_PATH

Convert to TensorFlow.js
bash tools/convert_tfjs_model.sh


See repository scripts for customization details.

📊 Monitoring Training

Use TensorBoard to visualize losses, images, and attention maps:

tensorboard --logdir=LOG_DIR

🧠 Technical Overview
Architecture

Generator:

Attentive recurrent network using ConvLSTM

Contextual autoencoder for detailed restoration

Discriminator:

Standard CNN-based discriminator

Attention Mechanism:

Highlights and localizes rain/raindrop regions

Loss Functions:

Multi-scale loss

Perceptual loss

Adversarial loss

📁 Script Overview
Script	Description
tools/train_model.py	Main training entry point
tools/test_model.py	Run inference on images
data_provider/data_feed_pipline.py	Convert datasets to TFRecords
tools/export_tf_saved_model.py	Export model to SavedModel
tools/convert_tfjs_model.sh	Convert model to TensorFlow.js
🗂️ Configuration

Main configuration file:

global_configuration/config.py


Contains:

Learning rate

Epochs

Batch size

Dataset + checkpoint paths

🖼️ Inference Output

The inference script produces:

Derained image

Attention map (visualizing rain regions)

📚 Reference

Attentive Generative Adversarial Network for Raindrop Removal from a Single Image
CVPR, 2018.
