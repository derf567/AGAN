User Manual
Overview
The "Attentive GAN DerainNet" is a Python-based deep learning toolkit for removing rain and raindrop artifacts from single images using a Generative Adversarial Network (GAN) with attention mechanisms. It is built on top of TensorFlow and is based on the CVPR 2018 paper "Attentive Generative Adversarial Network for Raindrop Removal from a Single Image".​

Installation
Requirements:

Python (preferably 3.5+)

TensorFlow (1.15 recommended)

CUDA and cuDNN for GPU support

Other dependencies listed in requirements.txt

Steps:

Clone the repo:

text
git clone https://github.com/derf567/attentive-gan-derainnet-
cd attentive-gan-derainnet-
Install dependencies:

text
pip install -r requirements.txt
(Optional) Set up CUDA and cuDNN for GPU acceleration.

How to Use
Testing (Inference)
To run the deraining model on a test image:

text
python tools/test_model.py --weights_path ./weights/derain_gan/derain_gan.ckpt-xxxxxx --image_path PATH_TO_IMAGE
Replace PATH_TO_IMAGE with your test image file path.

Replace weights path with your trained/checkpointed weights.​

Training
Prepare Data:

Organize your dataset into two folders:

rain_image/ : Images with rain

clean_image/ : Ground-truth clean images​

Convert to TFRecords:

text
python data_provider/data_feed_pipline.py --dataset_dir DATASET_ROOT --tfrecords_dir TFRECORDS_SAVE_DIR
Train the Model:

text
python tools/train_model.py --dataset_dir DATASET_ROOT
Training parameters (epochs, batch size, learning rate) can be set in global_configuration/config.py.​

Exporting
To export the model to TensorFlow SavedModel or TFJS:

text
python tools/export_tf_saved_model.py --export_dir SAVE_DIR --ckpt_path CKPT_PATH
bash tools/convert_tfjs_model.sh
Refer to repo scripts for details.​

Monitoring Training
Training progress can be monitored using TensorBoard.​

Technical Manual
Architecture
Generator: Attentive-recurrent network (using ConvLSTM), Contextual Autoencoder

Discriminator: Standard CNN-based network

Attention map: Helps to locate and highlight rain regions for effective deraining

Loss functions: Multi-scale loss, perceptual loss, adversarial loss

Scripts Overview
train_model.py: Model training entry point.

test_model.py: Model inference/test entry point.

data_feed_pipline.py: Data conversion to TFRecords for efficient feeding.

export_tf_saved_model.py: Save trained model for deployment.

convert_tfjs_model.sh: Converts TensorFlow model to JS format for web usage.

Configuration
Most model and training parameters are found in global_configuration/config.py.​

Learning rate, epochs, batch size

Paths for checkpoints and datasets

Dataset Preparation
Training pairs should be of matching resolution, typically organized under rain_image/ and clean_image/.

Automatic scaling and TFRecords conversion.​

Model Training
Default batch size: 1 (removes need for Batch Normalization).​

Recommended initial learning rate: 0.002.

Training is typically performed for over 100,000 iterations.

Inference Procedure
Loads pre-trained weights and applies the deraining GAN to input images for restoration.

Outputs both derained images and attention maps visualizing rain localization.

Export and Deployment
Models can be exported to the standard TensorFlow SavedModel format for serving.

Optionally, conversion to TensorFlow.js is supported for browser inference.​

References
Attentive Generative Adversarial Network for Raindrop Removal from A Single Image (CVPR 2018).​
