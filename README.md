
# Real-ESRGAN Image Upscaler

A Flask web application implementation of Real-ESRGAN (Real-World Single Image Super-Resolution) for high-quality image upscaling. This model shows better results on faces compared to the original version and provides both a web interface and Python API for easy integration.

> This is not an official implementation. We partially use code from the [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images.

## Features

- 📈 Up to 4x upscaling of images
- 🖼️ Supports common image formats (PNG, JPEG, etc.)
- 🌐 Easy-to-use web interface
- 🐍 Python API for integration
- 🚀 GPU acceleration support (CUDA)
- 💻 CPU fallback when GPU is not available
- 👤 Enhanced results for face images

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Web Interface

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`
3. Upload your image using the interface
4. Wait for processing to complete
5. Download the enhanced image

### Python API Usage

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

## Project Structure

```
Real-ESRGAN/
├── app.py              # Flask web application
├── requirements.txt    # Python dependencies
├── RealESRGAN/        # Main package directory
│   ├── __init__.py
│   ├── model.py       # Model implementation
│   ├── utils.py       # Utility functions
│   └── rrdbnet_arch.py# Network architecture
├── static/            # Web interface assets
│   └── index.html     # Web interface
└── weights/           # Model weights directory
```

## Examples

### Standard Image Enhancement
Low quality image:
![](inputs/lr_image.png)

Real-ESRGAN result:
![](results/sr_image.png)

### Face Enhancement
Low quality image:
![](inputs/lr_face.png)

Real-ESRGAN result:
![](results/sr_face.png)

### Detail Enhancement
Low quality image:
![](inputs/lr_lion.png)

Real-ESRGAN result:
![](results/sr_lion.png)

## Try it Online

You can try it in [Google Colab](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing)

## References

- [Paper: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Hugging Face 🤗 Model](https://huggingface.co/sberbank-ai/Real-ESRGAN)

## Requirements

- Python 3.7+
- PyTorch >= 1.7.0
- CUDA-compatible GPU (optional, for faster processing)
- See requirements.txt for detailed package dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Based on the [original Real-ESRGAN paper](https://arxiv.org/abs/2107.10833)
- Model weights provided by [Hugging Face](https://huggingface.co/sberbank-ai/Real-ESRGAN)
```

This is the complete updated version of the README.md file that includes all the necessary information about the project, including:
- Project description and features
- Installation and usage instructions (both web interface and Python API)
- Project structure
- Examples with images
- Online demo link
- References
- Requirements
- License and contributing guidelines
- Acknowledgments

The file is now properly formatted in Markdown and contains all the essential information for users and contributors.

        
