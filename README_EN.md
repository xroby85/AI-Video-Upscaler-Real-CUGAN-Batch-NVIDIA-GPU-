# 🎬 AI Video Upscaler — Real-CUGAN Batch (NVIDIA GPU)

> Automatically upscales low-resolution videos to **Full HD (1920×1080)** using AI,
> with batch processing on NVIDIA RTX GPUs.

---

## ✨ What it does

- Select a **folder containing `.mp4` files**
- The script automatically processes **all videos**, including subfolders
- Each video is upscaled to **1920×1080** using an AI super-resolution model
- The result is saved as `original_name_mastered.mp4` next to the source file
- Already processed files are **automatically skipped** (no overwriting)

---

## 🧠 How it works — Per-frame Pipeline

Every video frame goes through a 5-stage pipeline:

```
Original frame (e.g. 480p)
        │
        ▼
  [1] Real-CUGAN AI 4×
        │  Convolutional neural network (SE model)
        │  Runs on NVIDIA GPU via ncnn
        ▼
  [2] Lanczos Resize ×1.30
        │  Additional scaling for extra sharpness
        │  High-quality interpolation (Lanczos4)
        ▼
  [3] Unsharp Mask (USM)
        │  GaussianBlur + addWeighted
        │  Compensates for AI-introduced softness
        ▼
  [4] Final Resize → 1920×1080
        │  Normalize to target resolution
        ▼
  [5] FFmpeg — H.264 re-encode + audio copy
        │  libx264, CRF 18 (high quality)
        │  Audio copied without re-encoding
        ▼
   name_mastered.mp4
```

### Why this pipeline?

| Stage | Reason |
|---|---|
| Real-CUGAN AI | Recovers real details (textures, edges) that a simple resize cannot reproduce |
| Extra ×1.30 Resize | Adds apparent resolution with no extra AI cost |
| Unsharp Mask | The human brain perceives sharper images as more detailed and higher quality |
| Lanczos4 | Best interpolation algorithm for image upscaling |
| FFmpeg CRF 18 | High-quality H.264 compression, compatible with any media player |

---

## ⚙️ Requirements

### Hardware
- NVIDIA GPU (recommended: RTX 20xx / 30xx / 40xx)
- Minimum 6 GB VRAM
- **NVIDIA Driver version 512.96 (cu124) or newer**

### Software
```bash
pip install realcugan-ncnn-py
pip install opencv-python
pip install tqdm
pip install numpy
```

- **FFmpeg** installed and available in PATH → [ffmpeg.org](https://ffmpeg.org/download.html)
- Python 3.8+

---

## 🚀 Usage

```bash
python AIupscaling1.py
```

1. Run the script
2. A dialog window appears → **select the folder** containing your `.mp4` files
3. Processing starts automatically for all found videos
4. Progress is displayed in the terminal via `tqdm` progress bar
5. When finished, `*_mastered.mp4` files will appear next to each original video

---

## 📁 Output structure

```
📂 your_folder/
 ├── video1.mp4               ← original (untouched)
 ├── video1_mastered.mp4      ← upscaled result ✅
 ├── video2.mp4
 ├── video2_mastered.mp4      ✅
 └── subfolder/
     ├── clip.mp4
     └── clip_mastered.mp4    ✅
```

---

## 🔧 GPU Configuration

In the code, `gpuid=1` forces the second GPU detected by the system (e.g. dedicated RTX).
If you have a single GPU, change it to `gpuid=0`:

```python
self.upscaler = Realcugan(
    gpuid=0,       # 0 = first GPU, 1 = second GPU
    scale=4,
    noise=0,
    model='se',
    tilesize=768,  # reduce if VRAM is insufficient (e.g. 512)
    ...
)
```

---

## 🛠️ Technologies used

| Technology | Role |
|---|---|
| [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | AI super-resolution model |
| [OpenCV](https://opencv.org/) | Video read/write, frame processing |
| [FFmpeg](https://ffmpeg.org/) | Final re-encoding + audio handling |
| [ncnn](https://github.com/Tencent/ncnn) | Cross-platform GPU inference |
| tkinter | GUI dialog for folder selection |
| tqdm | Terminal progress bar |

---

## 📌 Notes

- Processing speed depends on your GPU and source resolution. A 1-minute clip at 480p takes roughly 2–5 minutes on an RTX 3060.
- `tilesize=768` is optimal for 8+ GB VRAM. Lower to `512` or `256` for GPUs with less VRAM.
- The script never modifies original files — it always works on copies.

---

## 👤 Author

Created by **Sacuiu Robert** — contributions and feedback are welcome!
