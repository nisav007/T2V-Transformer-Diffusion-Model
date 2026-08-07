import os
import subprocess
import importlib.util
import sys
import cv2
import torch
from tqdm import tqdm
import argparse

# ============================================================
# CONFIG
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video_dir",
    type=str,
    required=True,
    help="Path to video directory"
)

parser.add_argument(
    "--latent_dir",
    type=str,
    required=True,
    help="Path to output latent embedding directory"
)
args = parser.parse_args()

LATENT_DIR=args.latent_dir
DEVICE = "cuda"
DTYPE = torch.bfloat16

REPO_DIR = "./LightX2V"

VAE_CKPT = "./models/vae/Wan2.1_VAE.pth"


# ============================================================
# INSTALL DEPENDENCIES
# ============================================================

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)


# ============================================================
# CLONE REPO
# ============================================================

if not os.path.exists(REPO_DIR):
    run_cmd(
        "git clone https://github.com/ModelTC/LightX2V.git"
    )


# ============================================================
# DOWNLOAD CHECKPOINT
# ============================================================

os.makedirs("./models/vae", exist_ok=True)

if not os.path.exists(VAE_CKPT):

    run_cmd(
        "hf download "
        "lightx2v/Autoencoders "
        "Wan2.1_VAE.pth "
        "--local-dir ./models/vae/"
    )


# ============================================================
# IMPORT WanVAE
# ============================================================
sys.path.insert(0, "/kaggle/working/LightX2V")
vae_file = os.path.join(
    REPO_DIR,
    "lightx2v/models/video_encoders/hf/wan/vae.py"
)

spec = importlib.util.spec_from_file_location(
    "wan_vae",
    vae_file
)

wan_vae = importlib.util.module_from_spec(spec)

spec.loader.exec_module(wan_vae)

WanVAE = wan_vae.WanVAE

print("WanVAE imported successfully")


# ============================================================
# LOAD MODEL
# ============================================================

model = WanVAE(
    vae_path=VAE_CKPT,
    dtype=DTYPE,
    device=DEVICE
)

print("Wan2.1 VAE loaded")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(args.latent_dir, exist_ok=True)


# ============================================================
# VIDEO LOADER
# ============================================================

def load_video(path):

    cap = cv2.VideoCapture(path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = torch.from_numpy(
            frame
        ).permute(2, 0, 1)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames found in {path}"
        )

    video = torch.stack(
        frames,
        dim=0
    )

    video = video.unsqueeze(0)

    return video


# ============================================================
# FIND VIDEOS
# ============================================================
VIDEO_DIR=args.video_dir
video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if (
        f.endswith(".mp4")
        and "reconstructed" not in f
    )
])

print(f"Total videos: {len(video_files)}")


# ============================================================
# ENCODE LOOP
# ============================================================

for video_file in tqdm(video_files):

    video_path = os.path.join(
        VIDEO_DIR,
        video_file
    )

    video_name = os.path.splitext(
        video_file
    )[0]

    latent_path = os.path.join(
        LATENT_DIR,
        f"{video_name}.pt"
    )

    if os.path.exists(latent_path):
        continue

    try:

        video = load_video(
            video_path
        )

        video = (
            video
            .to(DEVICE, DTYPE)
            .div_(255.0)
        )

        with torch.no_grad():

            latent = model.encode_video(
                video
            )

        if isinstance(latent, tuple):
            latent = latent[0]

        torch.save(
            latent.cpu(),
            latent_path
        )

    except Exception as e:

        print(
            f"FAILED: {video_file}"
        )

        print(e)

print("Done")
