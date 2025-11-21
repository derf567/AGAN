Dashcam Enhancement System - User Manual
Version 1.0
Institution: Davao City Research
Author: Dashcam Enhancement Research Team

Table of Contents

Introduction
System Overview
System Requirements
Installation Guide
Quick Start Guide
Feature Descriptions
Usage Instructions
Understanding Output
Troubleshooting
Frequently Asked Questions
Support and Contact


1. Introduction
1.1 About This Manual
This user manual provides comprehensive instructions for using the Dashcam Enhancement System. Whether you're a researcher, law enforcement officer, or dashcam user, this guide will help you enhance low-quality dashcam footage effectively.
1.2 Purpose
The Dashcam Enhancement System is designed to improve the quality of dashcam footage affected by:

Rain and water droplets
Headlight glare from oncoming vehicles
Low light conditions
Poor contrast and visibility

1.3 Who Should Use This Manual

End Users: Individuals who need to enhance dashcam footage
Researchers: Those conducting studies on image enhancement
Law Enforcement: Officers analyzing dashcam evidence
Fleet Managers: Organizations managing vehicle fleets with dashcams

1.4 Document Conventions

Bold text: Important terms and UI elements
Italic text: File names and paths
Code blocks: Commands and code examples
⚠️ Warning: Critical information
💡 Tip: Helpful suggestions
ℹ️ Note: Additional information


2. System Overview
2.1 What is the Dashcam Enhancement System?
The Dashcam Enhancement System is an AI-powered software solution that automatically detects and corrects quality issues in dashcam footage. It uses advanced neural network models (RetinexNet and Attentive-GAN) to:

Detect Issues: Automatically identify rain, glare, and low-quality conditions
Apply Corrections: Use appropriate enhancement techniques
Generate Reports: Provide detailed metrics about improvements

2.2 Key Features
Intelligent Detection

Automatic rain detection
Headlight glare identification
Quality assessment
Scene type detection (day/night)

Enhancement Capabilities

Deraining: Remove rain streaks and droplets
Deglaring: Reduce headlight and light source glare
General Enhancement: Improve contrast and visibility
Selective Processing: Apply enhancements only where needed

Metrics and Reports

PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
Contrast improvement measurements
Before/after comparisons

2.3 System Architecture
The system consists of three main components:

Intelligent Switch (intelligent_switch.py)

Analyzes input images
Determines appropriate processing mode
Routes images to correct enhancement module


Enhancement Modules

Deraining module (GAN-based)
Deglaring module (RetinexNet-based)
General enhancement module


Metrics Generator

Calculates quality metrics
Generates reports
Creates comparison visualizations




3. System Requirements
3.1 Hardware Requirements
Minimum Requirements

Processor: Intel Core i5 or AMD equivalent
RAM: 8 GB
GPU: Not required (CPU mode available)
Storage: 2 GB free space
Display: 1280x720 resolution

Recommended Requirements

Processor: Intel Core i7 or AMD Ryzen 7
RAM: 16 GB or more
GPU: NVIDIA GPU with 4GB VRAM (CUDA support)
Storage: 10 GB free space (for datasets)
Display: 1920x1080 resolution or higher

3.2 Software Requirements

Operating System:

Windows 10/11 (64-bit)
Linux (Ubuntu 18.04 or later)
macOS 10.15 or later


Python: Version 3.7.9 (as specified)
Required Libraries:

TensorFlow 1.x
OpenCV 4.x
NumPy
scikit-image
PyYAML
Matplotlib
Pandas



3.3 Supported Image Formats

Input: PNG, JPG, JPEG
Output: PNG, JPG
Recommended: PNG for best quality preservation


4. Installation Guide
4.1 Pre-Installation Checklist
Before installing, ensure you have:

 Python 3.7.9 installed
 Administrator/root access
 Internet connection (for downloading dependencies)
 Sufficient disk space (minimum 2 GB)

4.2 Step-by-Step Installation
Step 1: Download the System
bash# Clone from GitHub
git clone https://github.com/yourusername/dashcam-enhancement-system.git

# Navigate to directory
cd dashcam-enhancement-system
Step 2: Set Up Python Environment
For Windows:
bash# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate
For Linux/Mac:
bash# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
Step 3: Install Dependencies
bash# Install required packages
pip install -r requirements.txt
Step 4: Download Pre-trained Weights
Download the model weights from the releases page and place them in:
dashcam-enhancement-system/
└── weights/
    └── derain_gan/
        └── derain_gan.ckpt-100000
Step 5: Verify Installation
bash# Test the installation
python test_installation.py
✅ If you see "Installation successful!", you're ready to use the system.
4.3 Troubleshooting Installation
Common Issues
Issue: "Module not found" error

Solution: Ensure virtual environment is activated and run pip install -r requirements.txt again

Issue: TensorFlow installation fails

Solution: Install specific version: pip install tensorflow==1.15.0

Issue: GPU not detected

Solution: Install CUDA toolkit compatible with your TensorFlow version


5. Quick Start Guide
5.1 Your First Enhancement
This quick tutorial will walk you through enhancing your first dashcam image.
Step 1: Prepare Your Image

Place your dashcam image in an accessible folder
Note the full path to your image
Recommended: Name it something simple like test_image.png

Step 2: Run the Enhancement
Basic Command:
bashpython intelligent_switch.py --image "path/to/your/image.png" --output "enhanced_output.png"
Example:
bashpython intelligent_switch.py --image "C:\Dashcam\rainy_night.png" --output "enhanced_rainy_night.png"
Step 3: View Results
The system will:

Analyze your image (5-10 seconds)
Apply appropriate enhancements (10-30 seconds)
Save the enhanced image
Generate a metrics report

Look for these outputs:

enhanced_output.png - Your enhanced image
metrics/ folder - Contains detailed reports
debug/ folder - Contains detection visualizations

5.2 Understanding the Output
After processing, you'll see messages like:
[STEP 1: CLEAN DETECTION - PRIORITY CHECK]
  Result: NEEDS PROCESSING ✗
  Confidence: 65.0%
  Reason: Headlight glare detected

[STEP 2: HEADLIGHT GLARE DETECTION]
  Headlight Glare Score: 0.008500
  Affected pixels: 1234 (2.5%)

[STEP 3: PROCESSING MODE SELECTED]
  Mode: DEGLARE

✓ DEGLARE processing completed
5.3 Five-Minute Tutorial
Goal: Enhance a rainy dashcam image

Open Command Prompt/Terminal

Windows: Press Win+R, type cmd, press Enter
Mac/Linux: Open Terminal application


Navigate to System Folder

bash   cd path/to/dashcam-enhancement-system

Activate Environment

bash   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate

Run Enhancement

bash   python intelligent_switch.py --image "test_images/sample.png" --output "results/enhanced.png"

Check Results

Open results/enhanced.png
Review metrics/ folder for quality reports




6. Feature Descriptions
6.1 Intelligent Detection System
The system automatically analyzes your image to determine what enhancements are needed.
Clean Image Detection

Purpose: Skips processing if image is already good quality
How it works: Checks contrast, brightness, noise, and glare
Benefits: Saves processing time and preserves original quality

Rain Detection

Threshold: Configurable (default: 0.068)
Method: Analyzes edge density and texture patterns
Indicators: High edge density, streaking patterns

Headlight Glare Detection

Threshold: Configurable (default: 0.005)
Method: Identifies bright, desaturated regions in lower image portion
Features:

Position-based filtering (focuses on road level)
Size filtering (targets 20-2000 pixel regions)
Aspect ratio checking (excludes long streaks)



Scene Type Detection

Day/Night Classification: Based on average brightness
Threshold: 80 (configurable)
Purpose: Applies scene-appropriate enhancement settings

6.2 Enhancement Modules
Deraining Module
Uses Attentive-GAN neural network to remove rain effects.
Capabilities:

Remove rain streaks
Eliminate water droplets
Restore occluded details
Preserve image structure

Best For:

Heavy rain conditions
Light rain with visible droplets
Rain-streaked windshields

Processing Time: 10-30 seconds depending on GPU/CPU
Deglaring Module
Uses RetinexNet-based multi-scale enhancement.
Capabilities:

Reduce headlight glare
Balance illumination
Enhance visibility in glare areas
Preserve non-affected regions

Best For:

Night driving with oncoming headlights
Bright street lights
Reflections on wet roads

Processing Time: 5-15 seconds
General Enhancement Module
Applies RetinexNet for overall quality improvement.
Capabilities:

Increase contrast
Improve brightness balance
Enhance details
Reduce noise

Best For:

Low-light conditions
Foggy weather
Underexposed footage

6.3 Configuration System
The system uses YAML configuration files for customization.
Configuration Files

enhanced_deglare_config.yaml

Deglaring settings
Threshold values
Processing parameters


deraining_config.yaml

Deraining settings
Day/night configurations
Post-processing options



Key Configuration Options
Detection Thresholds:
yamlrain_detection:
  threshold: 0.068

glare_detection:
  threshold: 0.005
  brightness_threshold: 220
  saturation_threshold: 30
Enhancement Parameters:
yamlretinex_enhancement:
  scales: [15, 80, 250]
  weights: [0.4, 0.4, 0.2]
  gamma_correction: 0.75

7. Usage Instructions
7.1 Basic Usage
Command Structure
bashpython intelligent_switch.py [OPTIONS]
Required Arguments

--image: Path to input image
--output: Path for enhanced output

Optional Arguments

--config: Custom configuration file
--generate_summary: Generate CSV summary of all metrics

Examples
Basic Enhancement:
bashpython intelligent_switch.py --image "input.jpg" --output "output.png"
With Custom Config:
bashpython intelligent_switch.py --image "input.jpg" --output "output.png" --config "my_config.yaml"
Generate Summary:
bashpython intelligent_switch.py --image "input.jpg" --output "output.png" --generate_summary
7.2 Processing Single Images
Method 1: Command Line

Open terminal in project directory
Activate virtual environment
Run command with your image path
Check output folder for results

Method 2: Python Script
Create a script process_image.py:
pythonfrom intelligent_switch import ConfigManager, IntelligentImageProcessor

# Initialize
config = ConfigManager('enhanced_deglare_config.yaml')
processor = IntelligentImageProcessor(config)

# Process image
result = processor.process_image('input.jpg', 'output.png')

# Check result
print(f"Status: {result['final_status']}")
print(f"Mode: {result['processing_mode']}")
7.3 Batch Processing
To process multiple images:
Create batch script batch_process.py:
pythonimport os
import glob
from intelligent_switch import ConfigManager, IntelligentImageProcessor

# Setup
config = ConfigManager('enhanced_deglare_config.yaml')
processor = IntelligentImageProcessor(config)

# Process all images in folder
input_folder = "input_images"
output_folder = "enhanced_images"
os.makedirs(output_folder, exist_ok=True)

for image_path in glob.glob(f"{input_folder}/*.png"):
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"enhanced_{filename}")
    
    print(f"Processing: {filename}")
    result = processor.process_image(image_path, output_path)
    print(f"Status: {result['final_status']}\n")
Run with:
bashpython batch_process.py
7.4 Using the RetinexNet Module Directly
For general enhancement without detection:
bashpython dashcam_enhancer.py "input_image.jpg" "enhanced_images"
This processes the image using RetinexNet and generates:

Enhanced image
Reflectance component
Illumination component
Comparison visualization
Detailed metrics report

7.5 Deraining Specific Usage
Basic Deraining
bashpython tools/test_model.py --image_path "rainy_image.jpg" --weights_path "weights/derain_gan/derain_gan.ckpt-100000"
Force Day Settings
bashpython tools/test_model.py --image_path "rainy_day.jpg" --weights_path "weights/derain_gan/derain_gan.ckpt-100000" --force_scene day
Force Night Settings
bashpython tools/test_model.py --image_path "rainy_night.jpg" --weights_path "weights/derain_gan/derain_gan.ckpt-100000" --force_scene night
7.6 Advanced Configuration
Customizing Detection Thresholds
Edit enhanced_deglare_config.yaml:
yamlrain_detection:
  threshold: 0.08  # Increase for less sensitive rain detection

glare_detection:
  threshold: 0.01  # Increase to only detect strong glare
  brightness_threshold: 230  # Only detect very bright areas
Adjusting Enhancement Strength
yamlretinex_enhancement:
  gamma_correction: 0.85  # Higher = brighter (0.5-1.2)
  contrast_strength: 1.5   # Higher = more contrast (0.8-2.0)
Enabling/Disabling Features
yamlselective_deglaring:
  enabled: true              # Enable selective deglaring
  enhance_only_glare_areas: true  # Only enhance glare regions

8. Understanding Output
8.1 Output Files
After processing, the system generates multiple files:
Primary Output

enhanced_image.png: Your processed image

Metrics and Reports
Located in metrics/ folder:
For Deglaring:

{filename}_metrics.txt: Detailed quality metrics
deglaring_summary.csv: Summary of all processed images
{filename}_comparison.png: Side-by-side comparison

For Deraining:

{filename}_metrics.txt: Deraining metrics
{filename}_derainmetrics.txt: Additional rain-specific metrics
{filename}_comparison.png: Before/after comparison
deraining_summary.csv: Summary CSV

Debug Outputs
Located in debug/ folder:

glare_detection.png or headlight_detection.png: Detection visualization
original.png: Original image copy
enhanced.png: Enhanced image copy
comparison.png: Stacked comparison

8.2 Understanding Metrics
PSNR (Peak Signal-to-Noise Ratio)

Range: 20-50 dB (higher is better)
Good: > 25 dB
Excellent: > 30 dB
Meaning: Measures how much noise/distortion was introduced

SSIM (Structural Similarity Index)

Range: 0-1 (higher is better)
Good: > 0.80
Excellent: > 0.90
Meaning: Measures preservation of image structure

Contrast Improvement

Measured in: Standard deviation increase
Positive values: Contrast increased
Negative values: Contrast decreased (may indicate over-processing)

Glare Area Reduced

Range: 0-1 (percentage reduced)
Good: > 0.60 (60% of glare removed)
Excellent: > 0.80 (80% of glare removed)

8.3 Reading the Console Output
Example Output Explained
[STEP 1: CLEAN DETECTION - PRIORITY CHECK]
  Result: NEEDS PROCESSING ✗
  Confidence: 65.0%
  Reason: ✗ Issues detected: Headlight glare detected (0.0085)

Image was analyzed and determined to need processing
65% confidence in this assessment
Detected headlight glare above threshold

[STEP 2: HEADLIGHT GLARE DETECTION]
  Headlight Glare Score: 0.008500 (threshold: 0.005000)
  Affected pixels: 1234 (2.500%)

Glare score exceeds threshold
1,234 pixels affected (2.5% of image)

[STEP 3: PROCESSING MODE SELECTED]
  Mode: DEGLARE

System chose deglaring mode

✓ DEGLARE processing completed

Processing successful

8.4 Interpreting Reports
Sample Metrics Report
IMAGE ENHANCEMENT METRICS REPORT
================================

ORIGINAL IMAGE METRICS
contrast       : 45.2300
brightness     : 178.5600
saturation     : 95.3400

ENHANCED IMAGE METRICS
contrast       : 58.7800
brightness     : 142.3200
saturation     : 112.5600

IMPROVEMENTS
contrast_gain            : 13.5500
brightness_reduction     : 36.2400
saturation_gain          : 17.2200
glare_area_reduced       : 0.7850
ssim                     : 0.8923
psnr                     : 28.4500
Interpretation:

✅ Contrast increased by 13.55 (good improvement)
✅ Brightness reduced by 36.24 in glare areas (glare reduced)
✅ Saturation increased by 17.22 (more vivid colors)
✅ 78.5% of glare removed
✅ SSIM of 0.89 (excellent structure preservation)
✅ PSNR of 28.45 dB (good quality)


9. Troubleshooting
9.1 Common Issues and Solutions
Issue: "Module not found" Error
Symptoms:
ModuleNotFoundError: No module named 'cv2'
Solutions:

Ensure virtual environment is activated
Reinstall dependencies:

bash   pip install -r requirements.txt

Check Python version:

bash   python --version  # Should be 3.7.9
Issue: "Weights not found" Error
Symptoms:
Weights not found: weights/derain_gan/derain_gan.ckpt-100000
Solutions:

Download weights from releases page
Place in correct directory:

   weights/
   └── derain_gan/
       └── derain_gan.ckpt-100000

Check file permissions

Issue: Processing is Very Slow
Symptoms:

Processing takes several minutes per image

Solutions:

GPU Not Detected:

Install CUDA toolkit
Verify GPU with: nvidia-smi (Windows/Linux)


Reduce Image Size:

yaml   system:
     max_image_dimension: 512  # Reduce from 1024

Use CPU Mode:

yaml   system:
     use_gpu: false
Issue: Poor Enhancement Results
Symptoms:

Image looks worse after processing
Too much noise introduced

Solutions:

Check if image needed enhancement:

System may have incorrectly detected issues
Use original if quality was already good


Adjust thresholds:

yaml   glare_detection:
     threshold: 0.01  # Increase to be less sensitive

Try different enhancement settings:

yaml   retinex_enhancement:
     gamma_correction: 0.85  # More conservative
     contrast_strength: 0.9   # Less aggressive
Issue: "Dimension mismatch" Error
Symptoms:
ValueError: Image size mismatch
Solutions:

System auto-fixes this now (recent update)
If persists, ensure using latest version
Try resizing input image to standard size:

python   img = cv2.resize(img, (512, 512))
Issue: Out of Memory Error
Symptoms:
ResourceExhaustedError: OOM when allocating tensor
Solutions:

Reduce GPU memory usage:

yaml   system:
     gpu_memory_fraction: 0.5  # Reduce from 0.8

Process smaller batches:

Process one image at a time
Close other GPU-intensive applications


Switch to CPU mode:

yaml   system:
     use_gpu: false
9.2 Error Messages Guide
Error MessageMeaningSolution"Could not read image"Invalid image file or pathCheck file path and format"CUDA not available"GPU not detectedInstall CUDA or use CPU mode"Config file not found"Missing configurationProvide config path or use default"Output directory not writable"Permission issueCheck folder permissions"Invalid image format"Unsupported formatConvert to PNG or JPG
9.3 Performance Optimization
For Faster Processing

Use GPU acceleration:

yaml   system:
     use_gpu: true
     gpu_memory_fraction: 0.8

Reduce image size:

yaml   inference:
     target_width: 512
     target_height: 512

Disable unnecessary features:

yaml   logging:
     generate_visualizations: false
     metrics_calculation: false  # Only if you don't need metrics
For Better Quality

Use higher resolution:

yaml   inference:
     target_width: 1024
     target_height: 1024

Enable all post-processing:

yaml   postprocessing:
     apply_clahe: true
     sharpen: true
     bilateral_filter: true

Use conservative settings:

yaml   retinex_enhancement:
     gamma_correction: 0.75
     contrast_strength: 1.0
9.4 Getting Help
If you encounter issues not covered here:

Check log files: Look in logs/ folder for detailed error messages
Create issue on GitHub:

Go to repository Issues page
Click "New Issue"
Provide:

Error message
System information
Input image characteristics
Configuration file used




Contact support: See Section 11


10. Frequently Asked Questions
10.1 General Questions
Q: How long does processing take?
A: Processing time varies:

With GPU: 5-15 seconds per image
With CPU: 30-120 seconds per image
Depends on image size and enhancement type

Q: Can I process video files?
A: Not directly. You need to:

Extract frames from video using ffmpeg or similar tool
Process each frame individually
Reassemble into video

Q: What image sizes are supported?
A: Any size, but images are resized to 512x512 or 1024x1024 for processing. Original aspect ratio is maintained where possible.
Q: Does processing modify my original image?
A: No. Original images are never modified. Enhanced versions are saved as separate files.
10.2 Technical Questions
Q: Why does the system sometimes skip processing?
A: The system includes "Clean Detection" that identifies images already in good quality. This:

Saves processing time
Preserves original quality
Avoids unnecessary artifacts

You can adjust sensitivity in configuration:
yamlclean_detection:
  min_confidence_to_skip: 0.70  # Adjust 0.0-1.0
Q: How accurate is the detection?
A: Detection accuracy varies:

Rain detection: ~85-90% accuracy
Glare detection: ~90-95% accuracy
Clean image detection: ~75-85% accuracy

Q: Can I force a specific enhancement mode?
A: Not directly via intelligent_switch.py, but you can:

Use modules directly (dashcam_enhancer.py or test_model.py)
Adjust thresholds to force detection
Modify the code to add force flags

Q: Why do I get different results each time?
A: Possible reasons:

Random initialization in neural networks
Different configuration files
System resource availability affecting processing
Different versions of libraries

For consistent results, use fixed random seeds and same environment.
10.3 Configuration Questions
Q: What's the difference between day and night settings?
A: Night settings typically:

Apply more aggressive brightness enhancement
Use stronger noise reduction
Increase saturation boost
Adjust CLAHE parameters

Q: Should I enable CLAHE post-processing?
A: Enable CLAHE if:

✅ Image has low contrast
✅ Dark regions need enhancement
✅ Processing nighttime footage

Disable CLAHE if:

❌ Image already has good contrast
❌ You notice halo artifacts
❌ Processing daytime footage

Q: How do I balance quality vs. speed?
A: For speed:
yamlinference:
  target_width: 512
  target_height: 512

postprocessing:
  apply_clahe: false
  bilateral_filter: false
For quality:
yamlinference:
  target_width: 1024
  target_height: 1024

postprocessing:
  apply_clahe: true
  bilateral_filter: true
  sharpen: true
10.4 Results Questions
Q: How do I know if enhancement improved the image?
A: Check these indicators:

Visual inspection: Does it look better?
SSIM value: Should be > 0.80
PSNR value: Should be > 25 dB
Contrast improvement: Should be positive
Glare reduction: Should be > 0.6 (60%)

Q: What if enhancement makes image worse?
A: This can happen if:

Image was already good quality (system should detect this)
Settings are too aggressive
Wrong enhancement mode was selected

Solutions:

Use original image
Adjust configuration settings
Try different enhancement module directly

Q: Can I undo processing?
A: Original images are never modified. Simply use your original file.
10.5 Advanced Questions
Q: Can I train my own models?
A: Yes, see Technical Manual section on model training. You'll need:

Training dataset (paired rainy/clean images)
GPU with sufficient memory
Training scripts (train_model.py)

Q: How do I create a custom configuration?
A: Copy existing config file and modify:
bashcp enhanced_deglare_config.yaml my_custom_config.yaml
# Edit my_custom_config.yaml
# Use with --config flag
Q: Can I integrate this into my application?
A: Yes, see Technical Manual for API usage:
pythonfrom intelligent_switch import IntelligentImageProcessor, ConfigManager

config = ConfigManager('config.yaml')
processor = IntelligentImageProcessor(config)
result = processor.process_image(input_path, output_path)
