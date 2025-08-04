# VideoScan
The official implementation of VideoScan+. which is a novel online video inference framework that achieves extremely reduced latency and memory usage.

<!-- <p align="center">
    <img src="assets/logo.png" width="400"/>
</p> -->

## 🌟 Highlights

- **Ultra-Low Latency**: Optimized streaming inference pipeline
- **Memory Efficient**: Significantly reduced memory footprint compared to traditional approaches
- **Online Processing**: Real-time video processing capabilities (around 50 serving FPS)
- **Easy Integration**: Simple integration with existing video processing pipelines

## 📅 **Timeline**

[**NEW!** 2025.08.05]: VideoScan+ official implementation is released! 🚀🚀🚀

## 🧐 Quick Start

### Installation

```bash
cd VideoScan
pip install -r requirements.txt
```

### Basic Usage

1. **Offline Video Inference**:
```python
python infer.py
```

1. **Online Streaming Video Inference**:
```python
python stream_infer.p
```

### Training

1. Stage 1 Training:
```bash
bash scripts/train_stg1.sh
```

2. Stage 2 Training:
```bash
bash scripts/train_stg2.sh
```


<!-- ## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{videoscan2024,
  title={VideoScan: Ultra-efficient Online Video Inference},
  author={Authors},
  journal={arXiv preprint},
  year={2024}
}
``` -->

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).