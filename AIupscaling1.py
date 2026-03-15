#!/usr/bin/env python3
"""
Video Upscaler Real-CUGAN - NVIDIA RTX EDITION - Batch Folder
Procesează toate .mp4 din folder → nume_mastered.mp4
"""

import os
import time
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tqdm import tqdm

print("🔍 Verificare Real-CUGAN...\n")

REALCUGAN_OK = False
try:
    from realcugan_ncnn_py import Realcugan
    REALCUGAN_OK = True
    print("✅ Real-CUGAN disponibil")
except ImportError:
    print("❌ Real-CUGAN nu este instalat")
    print("   Instalează cu: pip install realcugan-ncnn-py")

if not REALCUGAN_OK:
    input("\nApasă Enter...")
    exit(1)


class NvidiaRealCuganBatchUpscaler:
    def __init__(self):
        self.input_folder = None
        self.upscaler = None
        self.fps = 30

    def setup_model(self):
        print("\n🚀 Inițializare NVIDIA RTX (Real-CUGAN Fast)...")
        try:
            self.upscaler = Realcugan(
                gpuid=1,           # <--- FORTARE NVIDIA
                scale=4,
                noise=0,
                model='se',
                tilesize=768,
                tta_mode=False,
                num_threads=8,
                syncgap=3
            )
            print(f"✅ GPU NVIDIA Activat! (Model: SE, gpuid=1)")
            return True
        except Exception as e:
            print(f"❌ Eroare NVIDIA: {e}")
            return False

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        self.input_folder = filedialog.askdirectory(
            title="Selectează folderul cu videouri (.mp4)"
        )
        root.destroy()
        
        if not self.input_folder:
            return False
        
        print(f"\n📁 Folder ales: {self.input_folder}")
        return True

    def upscale_frame(self, frame):
        try:
            # Pas 1: Real-CUGAN 2× → ~960p
            result = self.upscaler.process_cv2(frame)
            
            # Pas 2: Resize ceva mai agresiv decât 2× total (efect ~2.3–2.5×)
            h, w = result.shape[:2]
            target_h = int(h * 1.30)   # 1.25–1.35 → testează; 1.30 e adesea sweet spot
            target_w = int(w * 1.30)
            result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pas 3: Sharpening final (USM – unsharp mask) ca să compensezi moalețea
            # Acesta face diferența mare în percepție fără cost AI
            blurred = cv2.GaussianBlur(result, (0, 0), 3.0)
            sharpened = cv2.addWeighted(result, 1.6, blurred, -0.6, 0)  # 1.6 / -0.6 = sharp decent
            # Limitează ca să nu apară halouri prea urâte
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Pas 4: Forțează 1920×1080 (crop sau pad dacă e nevoie, dar de obicei resize ok)
            final = cv2.resize(sharpened, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
            
            return final
        except Exception as e:
            print(f"⚠️ Eroare frame: {e}")
            return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

    def should_process(self, path: Path) -> bool:
        name = path.name.lower()
        if not name.endswith('.mp4'):
            return False
        if '_mastered' in name:
            return False
        return True

    def get_output_path(self, input_path: Path) -> Path:
        return input_path.with_name(input_path.stem + "_mastered.mp4")

    def process_single_video(self, input_path: Path):
        output_path = self.get_output_path(input_path)
        
        # skip dacă există deja
        if output_path.exists():
            print(f"⏭️  Skip (deja există): {output_path.name}")
            return

        print(f"\n🎥 Procesez: {input_path.name}")
        print(f"   → {output_path.name}")

        start_time = time.time()
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print("   ❌ Eroare la deschidere video")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_path.with_name(output_path.stem + "_temp.mp4")
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (1920, 1080))

        with tqdm(total=total_frames, unit="frame", desc=input_path.name[:40]) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                upscaled = self.upscale_frame(frame)
                out.write(upscaled)
                pbar.update(1)

        cap.release()
        out.release()

        # Mutăm audio + re-encode video ușor
        print("   🔊 Copiere audio + finalizare...")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(temp_output),
                '-i', str(input_path),
                '-map', '0:v:0', '-map', '1:a:0?',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(output_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if temp_output.exists():
                temp_output.unlink()
            print(f"   ✅ Gata în {int(time.time() - start_time)} sec")
        except Exception as e:
            print(f"   ⚠️ Problema cu ffmpeg: {e}")
            if temp_output.exists():
                temp_output.rename(output_path)
            print("   → salvat varianta fără audio optimizat")

    def process_folder(self):
        if not self.input_folder:
            return

        video_files = [
            p for p in Path(self.input_folder).glob("**/*.mp4")
            if self.should_process(p)
        ]

        if not video_files:
            print("\n❌ Nu am găsit fișiere .mp4 valide de procesat (fără _mastered deja).")
            return

        print(f"\n📊 Am găsit {len(video_files)} videouri de procesat:\n")
        for f in video_files:
            print(f"  • {f.name}")

        print("\n" + "="*70)
        print("  ⚡ Încep procesarea batch ...")
        print("="*70 + "\n")

        for video_path in video_files:
            self.process_single_video(video_path)

        print("\n" + "═"*70)
        print("🎉 Procesare batch terminată!")
        print("═"*70)


if __name__ == "__main__":
    app = NvidiaRealCuganBatchUpscaler()
    if app.setup_model():
        if app.select_folder():
            app.process_folder()
            input("\nApasă Enter pentru a închide...")