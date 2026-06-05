#  👀 [ICML 2026] Attend to Anything Model (AAM) 
> **The official implementation of [Attend to Anything: Foundation Model for Unified Human Attention Modeling (Saliency Prediction)](https://icml.cc/virtual/2026/poster/61103)**


<div align="center">
  
[![Paper](https://img.shields.io/badge/📜_Paper-ArXiv-red)](https://arxiv.org/abs/2606.03540)
[![License](https://img.shields.io/badge/📄_License-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/🐍_Python-3.10+-blue)]()
[![CN-Version](https://img.shields.io/badge/🇨🇳_Paper-中文版-blue)](https://github.com/wz-zhao/Attend-to-Anything/blob/main/AAM_ICML2026_Chinese.pdf)

</div>

---

<div align="center">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/actioncliptest00001_2_pred_overlay.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/actioncliptest00005_3_pred_overlay.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/actioncliptest00006_3_pred_overlay.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Diving-Side-001-Diving-Side-004__Golf-Swing-Back-001-Golf-Swing-Back-004.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Golf-Swing-Back-005-Lifting-002__Riding-Horse-001-Riding-Horse-004.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Run-Side-001-Run-Side-004__SkateBoarding-Front-001-SkateBoarding-Front-004.gif?raw=true" width="16%">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Swing-Bench-001-Swing-Bench-004__Swing-Bench-005-Swing-SideAngle-002.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Swing-SideAngle-003-Walk-Front-002__Walk-Front-003-Walk-Front-005.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V19_Singing1_snip_000016-V20_Singing2_snip_000019__V20_Singing2_snip_000020-V20_Singing2_snip_000023.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V21_Tap1_snip_000032-V26_Piano2_snip_000035__V26_Piano2_snip_000036-V28_Dog1_snip_000039.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V40_Guitar5_snip_000056-V40_Guitar5_snip_000059__V42_Violin2_snip_000060-V42_Violin2_snip_000063.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V7_Basketball2_snip_000088-V7_Basketball2_snip_000091__V7_Basketball2_snip_000092-V7_Basketball2_snip_000092.gif?raw=true" width="16%">
</div>


## 🌟 The first Unified Foundation Model for Attention Modeling (Saliency)

<div align="center">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/fig_intro.png?raw=true" height="260">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/fig_data.png?raw=true" height="260">
</div>


-  Human attention (saliency prediction) selectively focuses on informative regions when perceiving images, videos, and audio-visual scenes. However, **existing attention and saliency models are often tailored to specific scenarios**, such as static images, videos, or isolated audio-visual settings, limiting their generalization in real-world applications.

-  We propose the **Attend to Anything Model (AAM)**, a unified foundation model for attention understanding across images, videos, and audio-visual scenes. AAM formulates attention as a **cognitive entailment relationship**, enabling prompt-driven reasoning over both salient regions and hierarchical task relationships. To connect static and dynamic attention, AAM further models video attention evolution as a temporally diffusive process inspired by fluid dynamics.

-  Experiments on **16 public benchmarks** show that AAM outperforms state-of-the-art methods by about **6%** on diverse attention and saliency tasks, while achieving a **4× speedup** in video inference. These results demonstrate AAM’s generality and efficiency as a foundation model for human attention prediction.


## ✨ Highlights

<table>
  <tr>
    <td align="center"><b>🌐 Unified Foundation Model</b></td>
    <td>AAM unifies human attention modeling across <b>images, videos, and audio-visual scenes</b>.</td>
  </tr>
  <tr>
    <td align="center"><b>🧠 Cognitive Entailment Learning</b></td>
    <td>AAM models attention as a <b>general-to-specific cognitive entailment process</b> in hyperbolic space.</td>
  </tr>
  <tr>
    <td align="center"><b>🌊 Fokker–Planck Dynamics</b></td>
    <td>AAM <b>bridges static and dynamic attention</b> through continuous temporal attention evolution.</td>
  </tr>
  <tr>
    <td align="center"><b>📦 Attention-1.75M Corpus</b></td>
    <td>A standardized large-scale corpus with <b>1.75M fixation instances</b> across diverse scenarios.</td>
  </tr>
</table>


## 🧩 Framework Overview

<div align="center">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/fig_overview.png?raw=true" width="95%">
</div>


## Code Usage

### Repository Layout

```text
.
??? train.py              # Main training entry
??? ablation.py           # Prompt ablation training/evaluation entry
??? dpt.py                # AAM DPT model and decoder head
??? Video_NSA.py          # Temporal video module used by DPTHead
??? audio_backbone.py     # Audio encoder and audio-visual fusion
??? dataloader_all.py     # Image/video/audio dataloader builders
??? data_process_uni.py   # Static image datasets
??? dataset_video.py      # Video datasets
??? dataset_audio.py      # Audio-video datasets
??? function.py           # Training, validation, and saliency export loops
??? loss.py               # Saliency losses and metrics
??? prompts.py            # Dataset prompt templates shared by train/ablation
??? training_utils.py     # DDP, checkpoint, prompt embedding, and optimizer helpers
??? dinov3/               # Local DINOv3 dependency
```

### Installation

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install wav2clip
```

The current training entry is configured for NPU distributed training. If you run on CUDA or CPU, review `setup_distributed_npu()` in `training_utils.py` before launching training.

### Data and Checkpoints

Large files are intentionally excluded from git. Put datasets and checkpoints in local folders, then pass their paths through command-line arguments or environment variables.

```bash
# Linux/macOS
export A2_DATA_ROOT=/path/to/datasets
export A2_AUDIO_ROOT=/path/to/datasets/Audio

# Windows PowerShell
$env:A2_DATA_ROOT="D:\datasets\AAM"
$env:A2_AUDIO_ROOT="D:\datasets\AAM\Audio"
```

Default locations used by the code:

| Item | Default / Argument |
| :-- | :-- |
| Dataset root | `datasets/` or `--data_root` |
| Audio dataset root | `<A2_DATA_ROOT>/Audio` or `A2_AUDIO_ROOT` |
| DINOv3 checkpoint | `--dino_ckpt` |
| Local DINOv3 repo | `--repo_dir ./dinov3` |
| Training outputs | `--save_root ./runs` |

### Training

```bash
python train.py \
  --data_root /path/to/datasets \
  --repo_dir ./dinov3 \
  --dino_ckpt ./web_pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dino_size l \
  --save_root ./runs \
  --batch_size 32 \
  --num_workers 8
```

## 📂 Datasets

### Training Datasets Used in AAM

AAM is trained on a unified collection of **image-based**, **video-based**, and **audio-visual** saliency datasets, covering diverse scenarios such as natural scenes, web pages, e-commerce, movies, sports, and multi-modal audio-visual scenes.

| Modality           | Dataset     | Publication                | Domain         |         Scale | Resolution  |  Frames |
| :----------------- | :---------- | :------------------------- | :------------- | ------------: | :---------- | ------: |
| 🖼️ Image          | SALICON     | CVPR 2015                  | Natural scenes | 15,000 images | 640 × 480   |       - |
| 🖼️ Image          | MIT1003     | ICCV 2009                  | Natural scenes |  1,003 images | Varied      |       - |
| 🖼️ Image          | CAT2000     | arXiv 2015                 | Natural scenes |  2,000 images | 1080 × 1920 |       - |
| 🖼️ Image          | OSIE        | Journal of Vision 2014     | Natural scenes |    700 images | 800 × 600   |       - |
| 🖼️ Image          | FIGRIM      | Vision Research 2015       | Natural scenes |  2,787 images | 1366 × 768  |       - |
| 🖼️ Image          | U-EYE       | ACM CHI 2023               | Web pages      |  1,583 images | Varied      |       - |
| 🖼️ Image          | FiWI        | ECCV 2014                  | Web pages      |    149 images | 1366 × 768  |       - |
| 🖼️ Image          | SalECI      | CVPR 2022                  | E-commerce     |    871 images | 720 × 720   |       - |
| 🎞️ Video          | DHF1K       | CVPR 2018                  | Natural scenes |  1,000 videos | 640 × 360   | 582,605 |
| 🎞️ Video          | Hollywood-2 | TPAMI 2015                 | Movies         |  1,707 videos | 720 × 480   | 487,207 |
| 🎞️ Video          | UCF Sports  | TPAMI 2015                 | Sports         |    150 videos | 720 × 480   |   9,900 |
| 🎞️ Video          | LEDOV       | ECCV 2018                  | Natural scenes |    538 videos | 1280 × 720  | 179,336 |
| 🔊🎞️ Audio-Visual | DIEM        | Cognitive Computation 2011 | Movies         |     84 videos | 1280 × 720  | 240,452 |
| 🔊🎞️ Audio-Visual | Coutrot-1   | Journal of Vision 2012     | People         |     60 videos | 1280 × 720  |   9,564 |
| 🔊🎞️ Audio-Visual | Coutrot-2   | IJCV 2014                  | Natural scenes |     40 videos | 1280 × 720  |  25,223 |
| 🔊🎞️ Audio-Visual | ETMD        | SPIC 2020                  | Movies         |     30 videos | 1280 × 720  | 109,788 |
| 🔊🎞️ Audio-Visual | SumMe       | ECCV 2014                  | Sports         |     25 videos | 640 × 360   |  52,744 |

### Training Dataset Download

<div align="center">

|           Modality           |                                                                                                  Download                                                                                                 |
| :--------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      🖼️ Image Saliency     | [![Image Data](https://img.shields.io/badge/Download-Image%20Datasets-blue?style=for-the-badge\&logo=googledrive)](https://drive.google.com/file/d/1h3vnUYwzhic11CLOoTL_x0tkYkh3CbJr/view?usp=drive_link) |
|      🎞️ Video Saliency     |                           [![Video Data](https://img.shields.io/badge/Download-Video%20Datasets-orange?style=for-the-badge\&logo=github)](https://github.com/wenguanwang/DHF1K)                           |
| 🔊🎞️ Audio-Visual Saliency |                     [![Audio-Visual Data](https://img.shields.io/badge/Download-Audio--Visual%20Datasets-purple?style=for-the-badge\&logo=github)](https://github.com/oraclefina/MSPI)                    |

</div>


### Downloadable Collection of Saliency / Human Attention Datasets

#### 🌍 Saliency Dataset Zoo

<table>
  <tr>
    <td width="18%" align="center"><b>Category</b></td>
    <td width="82%" align="center"><b>Datasets</b></td>
  </tr>

  <tr>
    <td align="center"><b>🖼️ Image Saliency</b></td>
    <td>
      <a href="http://saliency.mit.edu/BenchmarkIMAGES.zip">MIT300</a> ·
      <a href="http://saliency.mit.edu/trainSet.zip">CAT2000</a> ·
      <a href="https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip">MIT1003</a> ·
      <a href="http://figrim.mit.edu/index_eyetracking.html">FIGRIM</a> ·
      <a href="https://github.com/NUS-VIP/saliency-in-crowd?tab=readme-ov-file">EyeCrowd</a> ·
      <a href="https://www-users.cse.umn.edu/~qzhao/webpage_saliency.html">FiWI</a> ·
      <a href="https://github.com/YueJiang-nj/UEyes-CHI2023">U-EYE</a> ·
      <a href="https://people.csail.mit.edu/tjudd/LowRes/">MIT Low-resolution</a> ·
      <a href="https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master">OSIE</a> ·
      <a href="https://live.ece.utexas.edu/research/doves/">DOVES</a> ·
      <a href="https://github.com/TsotsosLab/AIM">Toronto</a> ·
      <a href="https://opendatalab.org.cn/OpenDataLab/SALICON">SALICON</a> ·
      <a href="https://saliencydetection.net/dut-omron/">DUT-OMRON</a> ·
      <a href="https://sites.google.com/view/cocosearch/coco-freeview">COCOFreeview</a> ·
      <a href="https://osf.io/cn5yp/overview">DAEMONS</a> ·
      <a href="https://github.com/leafy-lee/E-commercial-dataset">SalECI</a>
    </td>
  </tr>

  <tr>
    <td align="center"><b>🎞️ Video Saliency</b></td>
    <td>
      <a href="https://github.com/wenguanwang/DHF1K">DHF1K</a> ·
      <a href="https://www.di.ens.fr/~laptev/actions/hollywood2/">Hollywood-2</a> ·
      <a href="https://www.crcv.ucf.edu/data/UCF_Sports_Action.php">UCF Sports</a> ·
      <a href="https://github.com/remega/LEDOV-eye-tracking-database?tab=readme-ov-file">LEDOV</a> ·
      <a href="http://ilab.usc.edu/bu/compress/">CRCNS</a> ·
      <a href="https://videoprocessing.ai/benchmarks/video-saliency-prediction.html">MSU Video Saliency Prediction</a>
    </td>
  </tr>

  <tr>
    <td align="center"><b>🔊🎞️ Audio-Visual Saliency</b></td>
    <td>
      <a href="http://antoinecoutrot.magix.net/public/databases.html">Coutrot Database 1</a> ·
      <a href="http://antoinecoutrot.magix.net/public/databases.html">Coutrot Database 2</a> ·
      <a href="https://videoprocessing.ai/benchmarks/video-saliency-prediction.html">SAVAM</a> ·
      <a href="https://github.com/MinglangQiao/Sports_saliency">Sports Saliency</a> ·
      <a href="https://challenges.videoprocessing.ai/challenges/video-saliency-prediction-participate.html">AViMoS</a>
    </td>
  </tr>

  <tr>
    <td align="center"><b>🚗 Driving Attention</b></td>
    <td>
      <a href="https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8">DRVE</a> ·
      <a href="https://github.com/JWFangit/LOTVS-DADA">DADA-2000</a> ·
      <a href="https://github.com/yuli1102/eye_tracker_data">Eye Tracker Data</a> ·
      <a href="https://paperswithcode.com/dataset/bdd-a">BDD-A</a>
    </td>
  </tr>
</table>

## ⚙️ Checkpoint

<div align="center">

[![Checkpoint](https://img.shields.io/badge/Download-AAM%20Checkpoint-red?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1ttIONm6Mzx2n5d1cqRXrq95hHbYT32Bt/view?usp=drive_link)

<br>

| Component | Source / Installation |
|:--:|:--|
| Visual Encoder | [DINOv3](https://github.com/facebookresearch/dinov3) |
| Text Encoder | `pip install git+https://github.com/openai/CLIP.git` |
| Audio Encoder | `pip install wav2clip` |

</div>


</div>

## 💥 Visual Results

<div align="center">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/fig_visual_app_01.png?raw=true" width="75%">
</div>


## 🗯️  Robustness experiment

<div align="center">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/fig_ab_2.png?raw=true" width="75%">
</div>


## 📚 Citation

If you find this repository useful, please use the following BibTeX entry for citation and give us a star⭐.

```bibtex
# BibTeX will be updated after the official citation metadata is finalized.
```
