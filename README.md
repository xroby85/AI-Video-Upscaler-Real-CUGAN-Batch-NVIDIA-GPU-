# 🎬 AI Video Upscaler — Real-CUGAN Batch (NVIDIA GPU)

> Upscalează automat videoclipurile cu rezoluție mică la **Full HD (1920×1080)** folosind AI,
> cu procesare în lot pe GPU-uri NVIDIA RTX.

---

## ✨ Ce face

- Selectează un **folder care conține fișiere `.mp4`**
- Scriptul procesează automat **toate videoclipurile**, inclusiv subfolderele
- Fiecare videoclip este scalat la **1920×1080** folosind un model de super-rezoluție AI
- Rezultatul este salvat ca `original_name_mastered.mp4` lângă fișierul sursă
- Fișierele deja procesate sunt **omite automat** (fără suprascriere)

---

## 🧠 Cum funcționează — Canalizare per cadru

Fiecare cadru video trece printr-un canal în 5 etape:

```
Cadru original (de exemplu, 480p)
│
▼
[1] Real-CUGAN AI 4×
│ Rețea neuronală convoluțională (model SE)
│ Rulează pe GPU NVIDIA prin ncnn
▼
[2] Redimensionare Lanczos ×1.30
│ Scalare suplimentară pentru claritate suplimentară
│ Interpolare de înaltă calitate (Lanczos4)
▼
[3] Mască de unsharp (USM)
│ GaussianBlur + addWeighted
│ Compensează pentru moliciunea introdusă de AI
▼
[4] Redimensionare finală → 1920×1080
│ Normalizare la rezoluția țintă
▼
[5] FFmpeg — Recodificare H.264 + copie audio
│ libx264, CRF 18 (calitate înaltă)
│ Audio copiat fără recodificare
▼
name_mastered.mp4
```

### De ce această pipeline?

| Etapă | Motiv |
|---|---|
| Real-CUGAN AI | Recuperează detalii reale (texturi, margini) pe care o simplă redimensionare nu le poate reproduce |
| Redimensionare suplimentară ×1.30 | Adaugă rezoluție aparentă fără costuri suplimentare AI |
| Mască de unsharp | Creierul uman percepe imagini mai clare ca fiind mai detaliate și de calitate superioară |
| Lanczos4 | Cel mai bun algoritm de interpolare pentru upscaling-ul imaginilor |
| FFmpeg CRF 18 | Compresie H.264 de înaltă calitate, compatibilă cu orice player media |

---

## ⚙️ Cerințe

### Hardware
- GPU NVIDIA (recomandat: RTX 20xx / 30xx / 40xx)
- Minim 6 GB VRAM
- **Driver NVIDIA versiunea 512.96 (cu124) sau mai nouă**

### Software
```bash
pip install realcugan-ncnn-py
pip install opencv-python
pip install tqdm
pip install numpy
```

- **FFmpeg** instalat și disponibil în PATH → [ffmpeg.org](https://ffmpeg.org/download.html)
- Python 3.8+

---

## 🚀 Utilizare

```bash
python AIupscaling1.py
```

1. Rulați scriptul
2. Apare o fereastră de dialog → **selectați folderul** care conține fișierele `.mp4`
3. Procesarea începe automat pentru toate videoclipurile găsite
4. Progresul este afișat în terminal prin intermediul barei de progres `tqdm`
5. Când ați terminat, fișierele `*_mastered.mp4` vor apărea lângă fiecare videoclip original

---

## 📁 Structura de ieșire

```
📂 folderul_tau/
├── video1.mp4 ← original (netuit)
├── video1_mastered.mp4 ← rezultat scalat ✅
├── video2.mp4
├── video2_mastered.mp4 ✅
└── subfolder/
├── clip.mp4
└── clip_mastered.mp4 ✅
```

---

## 🔧 Configurarea GPU

În cod, `gpuid=1` forțează al doilea GPU detectat de sistem (de exemplu, (RTX dedicat).
Dacă aveți un singur GPU, schimbați-l la `gpuid=0`:

```python
self.upscaler = Realcugan(
gpuid=0, # 0 = primul GPU, 1 = al doilea GPU
scale=4,
noise=0,
model='se',
tilesize=768, # reduceți dacă VRAM este insuficient (de exemplu, 512)
...
)
```

---

## 🛠️ Tehnologii utilizate

| Tehnologie | Rol |
|---|---|
| [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | Model de super-rezoluție AI |
| [OpenCV](https://opencv.org/) | Citire/scriere video, procesare cadre |
| [FFmpeg](https://ffmpeg.org/) | Recodificare finală + gestionare audio |
| [ncnn](https://github.com/Tencent/ncnn) | Inferență GPU multi-platformă |
| tkinter | Dialog GUI pentru selectarea folderului |
| tqdm | Bara de progres a terminalului |

---

## 📌 Note

- Viteza de procesare depinde de rezoluția GPU și a sursei. Un clip de 1 minut la 480p durează aproximativ 2-5 minute pe un RTX 3060.
- `tilesize=768` este optim pentru 8+ GB VRAM. Reduceți la `512` sau `256` pentru GPU-uri cu mai puțin VRAM.
- Scriptul nu modifică niciodată fișierele originale - funcționează întotdeauna pe copii.

---

## 👤 Autor
Creat de **Săcuiu Robert** — contribuțiile și feedback-ul sunt binevenite!
