
# T2V-Transformer-Diffusion-Model


> Transformer-based latent video diffusion framework with **Temporal Cross-Layer Attention (TCLA)** for improving temporal coherence, motion continuity, and object identity preservation in generated videos.

---

## Overview

This repository contains the implementation of **Temporal Cross-Layer Attention (TCLA)**, a temporal feature propagation framework for transformer-based latent video diffusion models.

The proposed architecture enhances temporal consistency by introducing a **Temporal Feature Cache** and **Temporal Cross-Layer Attention (TCLA)** inside the Diffusion Transformer (DiT). Instead of processing each frame independently, the model reuses motion-aware features extracted from previous frames and earlier transformer layers, allowing the network to model temporal dependencies while preserving spatial information.

The framework is evaluated on two benchmark datasets:

- **Vimeo-90K** for video frame interpolation.
- **MSR-VTT** for text-to-video generation.

---

## Architecture


<img width="1407" height="768" alt="Gemini_Generated_Image_wi0371wi0371wi03 (1)" src="https://github.com/user-attachments/assets/789ce937-583b-4326-8f28-7031eb2f0a37" />


The overall pipeline consists of:

1. Video preprocessing
2. LightX2V VAE latent encoding
3. CLIP text encoding
4. Latent patch embedding
5. Diffusion Transformer
6. Spatial Feature Cache
7. Temporal Feature Cache
8. Temporal Cross-Layer Attention (TCLA)
9. Cross-Attention with text
10. Noise prediction
11. Reverse diffusion
12. Video reconstruction

---

## Key Features

- Transformer-based latent video diffusion
- LightX2V VAE latent representation
- CLIP text conditioning
- Diffusion Transformer (DiT)
- Spatial Feature Cache
- Motion-aware Temporal Feature Cache
- Temporal Cross-Layer Attention (TCLA)
- Exponential Moving Average (EMA) training
- Classifier-Free Guidance (CFG)

---

## Repository Structure

```text
.
├── configs/
├── datasets/
├── models/
│   ├── dit.py
│   ├── dit_block.py
│   ├── attention.py
│   ├── cache.py
│   └── tcla.py
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Diffusion Transformer

| Parameter | Value |
|------------|------:|
| Transformer Depth | 6 |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Patch Size | 2 × 2 |
| Normalization | RMSNorm |
| Conditioning | AdaLN |
| Input Latent Shape | (12,16,28,36) |
| Text Embedding | (77,512) |

---

## Temporal Cross-Layer Attention

The proposed TCLA module operates after the self-attention layer of each Diffusion Transformer block.

For every frame:

- Current frame tokens act as **Query**
- Cached motion-aware tokens from previous frames act as **Key** and **Value**
- Cross-attention aggregates temporal information
- The attended features are fused using residual connections

This enables temporal feature propagation across both transformer layers and neighboring frames.

---

## Temporal Feature Cache

The Temporal Feature Cache stores compact motion-aware representations instead of all latent tokens.

Features:

- Motion-aware token selection
- Top-K informative tokens
- Layer-wise cache organization
- Frame-wise temporal retrieval
- Sliding temporal window
- Efficient memory utilization

---

## Datasets

### Vimeo-90K

Used for evaluating temporal feature propagation through **video frame interpolation**.

- 64,612 training clips
- 7,824 testing clips
- Seven-frame video sequences
- Three-frame interpolation setting

---

### MSR-VTT

Used for **text-to-video generation**.

- 10,000 videos
- 200,000 captions
- 20 captions per video
- 20 semantic categories

---

## Preprocessing Pipeline

```text
MSR-VTT Dataset
        │
        ▼
Video Preprocessing
        │
        ▼
Random Caption Selection
        │
        ▼
LightX2V VAE Encoder
        │
        ▼
Latent Representation
(12,16,28,37)
        │
        ▼
Patch Embedding
        │
        ▼
Diffusion Transformer
```

---

## Training Pipeline

```text
Latent Tokens
        │
        ▼
Self Attention
        │
        ▼
Spatial Feature Cache
        │
        ▼
Temporal Feature Cache
        │
        ▼
Temporal Cross-Layer Attention
        │
        ▼
Cross Attention (Text)
        │
        ▼
Feed Forward Network
        │
        ▼
Noise Prediction
        │
        ▼
MSE Loss
        │
        ▼
AdamW
        │
        ▼
EMA Update
```

---

## Training Configuration

| Parameter | Value |
|------------|------:|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Batch Size | 1 |
| Transformer Depth | 6 |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Patch Size | 2×2 |
| EMA Decay | 0.999 |

---

## Model Size

| Metric | Value |
|---------|------:|
| Total Parameters | 38,076,928 |
| Trainable Parameters | 38,076,928 |
| Model Size | 38.08 Million Parameters |

---

## Results

The proposed Temporal Cross-Layer Attention improves temporal feature propagation by:

- Preserving object identity across frames
- Enhancing motion continuity
- Improving temporal coherence
- Reusing motion-aware latent representations
- Maintaining efficient computational complexity

Quantitative and qualitative comparisons are provided for both Vimeo-90K and MSR-VTT benchmarks.

---

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Diffusers
- OpenCV
- NumPy
- Torchvision
- Accelerate

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{tcla2026,
  title={Temporal Cross-Layer Attention for Enhanced Video Diffusion},
  author={Nishant Sahu},
  school={National Institute of Technology Karnataka (NITK), Surathkal},
  year={2026}
}
```

---

## License

This project is released for academic and research purposes.
