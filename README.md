# Medical Image Segmentation Benchmark

A comprehensive comparison of state-of-the-art neural network architectures for medical image segmentation, specifically focused on polyp segmentation in colonoscopy images.

## 🔬 Overview

This project implements and compares 10 different deep learning models for medical image segmentation across 6 benchmark datasets. Each model is paired with its specialized loss function to optimize performance for polyp detection and segmentation tasks.

## 📊 Datasets

### Supported Datasets

#### 1. CVC-ClinicDB Dataset
- **Dataset Link**: https://polyp.grand-challenge.org/CVCClinicDB/
- **Path**: `datasets/CVC-ClinicDB/`
- **Images**: `Original/` folder (`.tif` format)
- **Masks**: `Ground Truth/` folder (`.tif` format)
- **Split**: 90% train, 10% test
- **Preprocessing**: Albumentations with advanced augmentations
- **Augmentations**: Horizontal/vertical flip, rotation, brightness/contrast, elastic transform

#### 2. Kvasir-SEG Dataset
- **Dataset Link**: https://datasets.simula.no/kvasir-seg/
- **Path**: `dataset/Kvasir-SEG/`
- **Images**: `images/` folder (`.jpg` format)
- **Masks**: `masks/` folder (`.jpg` format)
- **Split**: 90% train, 10% test
- **Preprocessing**: OpenCV-based with standard normalization


#### 3. Retina Blood Vessel Dataset
- **Dataset Link**: https://data.mendeley.com/datasets/frv89hjgrr/1
- **Dataset Link**: https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel
- **Path**: `dataset/Retina-Blood-Vessel/`
- **Train folder**:
- **Images**: `images/` folder (`.png` format)
- **Masks**: `masks/` folder (`.png` format)
- **Test folder**:
- **Images**: `images/` folder (`.png` format)
- **Masks**: `masks/` folder (`.png` format)
- **Preprocessing**: OpenCV-based with standard normalization

#### 4. ISIC (Skin Lesion) Dataset
- **Dataset Link**: https://challenge.isic-archive.com/data/#2017
- **Path**: `dataset/Skin-Lesion/`
- **Images**: `images/` folder (`.jpg` format)
- **Masks**: `masks/` folder (`.png` format)
- **Split**: 90% train, 10% test
- **Preprocessing**: OpenCV-based with standard normalization

#### 5. Breast Ultrasound Images Dataset
- **Dataset Link**: https://www.sciencedirect.com/science/article/pii/S2352340919312181
- **Dataset Link**: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download
- **Path**: `dataset/Breast-Ultrasound-Images/`
- **benign**: images and masks are in `.png` format
- **malignant**: images and masks are in `.png` format
- **normal**: images and masks are in `.png` format
- **Split**: 90% train, 10% test
- **Preprocessing**: OpenCV-based with standard normalization

#### 6. Brain Tumor Dataset
- **Dataset Link**: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=3381290
- **Dataset Link**: https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation?resource=download
- **Path**: `dataset/Brain-Tumor/`
- **Images**: `images/` folder (`.png` format)
- **Masks**: `masks/` folder (`.png` format)
- **Split**: 90% train, 10% test
- **Preprocessing**: OpenCV-based with standard normalization

### Image Specifications
- **Input Size**: 224×224 pixels
- **Format**: RGB (3 channels)
- **Normalization**: [0, 1] range
- **Batch Size**: 4 (configurable)

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch (with MPS/CUDA support)
- scikit-image
- albumentations
- OpenCV (cv2)
- scikit-learn
- pandas

### Dataset Setup

Organize your datasets as follows:

```
datasets/
├── CVC-ClinicDB/
│   ├── Original/          # .tif image files
│   └── Ground Truth/      # .tif mask files
└── Kvasir-SEG/
    ├── images/            # .jpg image files
    └── masks/             # .jpg mask files
```

### Hardware Requirements

- **Recommended**: Apple Silicon Mac (MPS support) or CUDA-capable GPU
- **Minimum**: CPU-only execution supported but slower
- **Memory**: 8GB+ RAM recommended

### Running the Benchmark

```bash
python main.py
```

This will automatically:
1. Detect your hardware (MPS/CUDA/CPU)
2. Load all datasets with preprocessing
3. Train all 10 models on both datasets
4. Apply dataset-specific augmentations
5. Save results and model checkpoints

## 📁 Project Structure

```
medical-segmentation-benchmark/
├── main.py                 # Main training script
├── train.py               # Training loop implementation
├── dataset.py             # Dataset loading and preprocessing
├── utils.py               # Helper functions and loss implementations
├── requirements.txt       # Python dependencies
├── models/                # Model architectures
│   ├── UneXt.py
│   ├── ESDMR_Net.py
│   ├── LEA_Net.py
│   ├── LTMSegNet.py
│   ├── LWBNA_Net.py
│   ├── LV_Net.py
│   ├── MLU_Net.py
│   ├── RetinaLite_Net.py
│   └── PMFS_Net.py
├── datasets/              # Dataset storage
│   ├── CVC-ClinicDB/
│   │   ├── Original/      # Original images (.tif)
│   │   └── Ground Truth/  # Segmentation masks (.tif)
│   └── Kvasir-SEG/
│       ├── images/        # Original images (.jpg)
│       └── masks/         # Segmentation masks (.jpg)
|---- ....
└── README.md             # This file
```


## 📈 Results and Evaluation

Results are automatically saved during training. The benchmark evaluates:

- **Segmentation Accuracy**: Pixel-wise accuracy
- **Dice Coefficient**: Overlap measure
- **IoU (Intersection over Union)**: Region overlap metric
- **Training Loss**: Model convergence tracking

## 🛠️ Customization

### Adding New Models

1. Create your model in `models/YourModel.py`
2. Import in `main.py`
3. Add to the models list with appropriate loss function

### Adding New Datasets

1. Implement dataset loader in `dataset.py`
2. Add dataset loading logic in `main.py`
3. Update the datasets list

### Modifying Loss Functions

Custom loss functions can be added to `utils.py` and imported in `main.py`.

## 🔍 Hardware Optimization

The project automatically optimizes for your hardware:

- **Apple Silicon (M1/M2/M3)**: Uses MPS backend for GPU acceleration
- **NVIDIA GPUs**: CUDA support (when available)
- **CPU Fallback**: For systems without GPU acceleration

## 📋 Requirements

Key dependencies include:
```
torch>=1.9.0
torchvision>=0.10.0
scikit-image>=0.18.0
albumentations>=1.0.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- New loss functions
- Dataset augmentation techniques
- Performance optimizations
- Evaluation metrics

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
[Coming soon]
```

## 🐛 Issues and Support

For issues, questions, or suggestions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include system information and error logs

---

**Note**: This benchmark is designed for research and educational purposes. For clinical applications, ensure proper validation and regulatory compliance.