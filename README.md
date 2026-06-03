#  👁 [ICML 2026] Attend to Anything: Foundation Model for Unified Human Attention Modeling (Saliency Prediction)


<p align="center">
<a href="https://arxiv.org/abs/2606.03540">
  <img src="https://img.shields.io/badge/arXiv-2602.01593-b31b1b.svg?logo=arxiv">
</a>


<!-- <p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_show_1.png" width="90%"> -->


<p align="center">
  Wenzhuo Zhao, Ronghao Xian, Keren Fu, Qijun Zhao
</p>

---


## 🚀 Introduction -  the first Foundation Model for Saliency Prediction (Attention Modeling)

-  Human attention naturally focuses on the most important regions when people view images and videos or perceive sounds. However, existing attention and saliency models are often **limited to specific scenarios or task settings**, such as static images, particular types of videos, or individual audio-visual contexts. As a result, they struggle to generalize flexibly in real-world applications.

-  We propose the Attend to Anything Model (AAM), a **unified foundation model** for understanding attention across images, videos, and audio-visual scenes. AAM **formulates attention as a cognitive entailment relationship**, enabling the model to reason not only about what is salient, but also about hierarchical task relationships from general to specific through language prompts. To bridge static image attention and dynamic video attention, we further draw inspiration from **fluid dynamics and model the temporal evolution of attention** in videos as a diffusion process over time.

-  Across 16 public benchmarks, AAM consistently outperforms existing state-of-the-art methods by an average of approximately **6%** on diverse attention and saliency tasks, while achieving about a **4× speedup** in video inference. These results suggest that AAM can serve as a more general and efficient foundation model for attention modeling, supporting future research on predicting where humans attend across image, video, and audio-visual tasks.

---

## 🔑 Motivation 

<!-- <p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_intro_1.png" width="70%">
</p>

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_sns_1.png" width="70%">
</p> -->

- 🧠 By rethinking human attention as a **unified hierarchical cognitive process**, AAM first unified image, video and audio-visual saliency prediction (attention modeling) across scenes and tasks.
- 🎯 
- 🌈 Support for **RGB SOD / RGB-D SOD / RGB-T SOD / VDT SOD / VSOD / RGB-D VSOD** via a **single versatile** model

---

## 🧩 Framework Overview

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_overview_1.png" width="85%">
</p>
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_overview_new_1.png" width="85%">
</p>

---

## 📂 Datasets

| Task | Training Datasets | Testing Datasets | Download |
|------|-------------------|------------------|----------|
| RGB SOD | DUTS | DUTS, ECSSD, HKU-IS, PASCAL-S, DUT-O | [Baidu](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g?pwd=x7kn) (`x7kn`) |
| RGB-D SOD | NJU2K, NLPR, DUT-RGBD | NJU2K, NLPR, DUT-RGBD, SIP, STERE | [Baidu](https://pan.baidu.com/s/1ibrO3CS7rn7bJUAy8hM9mQ?pwd=8b9c) (`8b9c`) |
| RGB-T SOD | VT5000 | VT5000, VT821, VT1000 | [Baidu](https://pan.baidu.com/s/1PKW5d_Yr5NFEnq9Q82HitA?pwd=xhrm) (`xhrm`) |
| VDT SOD | VDT-2048 | VDT-2048 | [Baidu](https://pan.baidu.com/s/1JyFBtjlJGf4GE2zeciN1wQ?pwd=bipy) (`bipy`) |
| VSOD | DAVIS, DAVSOD, FBMS | DAVIS, DAVSOD, FBMS, Seg-V2, VOS | [Baidu](https://pan.baidu.com/s/1zQ-vuDnSfRzJ1T_T-hh7sA?pwd=kcmu) (`kcmu`) |
| RGB-D VSOD | RDVS, DVisal, Vidsod_100 | RDVS, DVisal, Vidsod_100 | [Baidu](https://pan.baidu.com/s/1VRL3jk7AsQCkL26hwg1rZA?pwd=q9ty) (`q9ty`) |

### 🛠️ Overlapping Samples
To avoid data leakage and ensure fair training, we only retain the samples from DVisal together with their ground-truth annotations.
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_overlap_1.png" width="70%">
</p>

---

## ✨  Visual Results
All evaluated saliency maps are put here: [Baidu](https://pan.baidu.com/s/1Lv9P8JW3YyI6Ds76wUnXaQ?pwd=p2h2)(`p2h2`)
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_visual_1.png" width="100%">
</p>

### Other Tasks that Emphasize Spatial Continuity

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/other.jpg" width="100%">
</p>

---

## ⚙️ Environment Setup

- PyTorch 1.13.1
- CUDA 11.7
- VMamba-S backbone weights：[[baidu](https://pan.baidu.com/share/init?surl=SaEV237VCzSEn558gEBiXg)(zsxa)]
- Samba+ weights：[[baidu](https://pan.baidu.com/s/1w7n2FuEo0R1hD-JE1JT-3w?pwd=3xxz)(3xxz)]; 
[![Google Drive](https://img.shields.io/badge/Google-Drive-green?logo=google-drive)](https://drive.google.com/file/d/1S-8RV9vJT5VLuFcRQ8OS9GLEvf31ziCo/view?usp=sharing)

## 📚 Citation

If you find this repository useful, please use the following BibTeX entry for citation and give us a star⭐.

```bibtex

@InProceedings{He_2025_CVPR,
  author    = {He, Jiahao and Fu, Keren and Liu, Xiaohong and Zhao, Qijun},
  title     = {Samba: A Unified Mamba-based Framework for General Salient Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {25314--25324}
}

