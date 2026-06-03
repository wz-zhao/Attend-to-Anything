#  👀 Attend to Anything Model (AAM) 
> **The official implementation of [ICML 2026] [Attend to Anything: Foundation Model for Unified Human Attention Modeling (Saliency Prediction)](https://icml.cc/virtual/2026/poster/61103)**

<div align="center">
  
[![Paper](https://img.shields.io/badge/📜_Paper-ArXiv-red)](https://arxiv.org/abs/2606.03540)
[![License](https://img.shields.io/badge/📄_License-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/🐍_Python-3.10+-blue)]()
[![CN-Version](https://img.shields.io/badge/🇨🇳_Paper-中文版-blue)](https://github.com/wz-zhao/Attend-to-Anything/blob/main/AAM_ICML2026_Chinese.pdf)

</div>



---

<br>

<div align="center">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/actioncliptest00001_2_pred_overlay.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/actioncliptest00005_3_pred_overlay.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/actioncliptest00006_3_pred_overlay.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Diving-Side-001-Diving-Side-004__Golf-Swing-Back-001-Golf-Swing-Back-004.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Golf-Swing-Back-005-Lifting-002__Riding-Horse-001-Riding-Horse-004.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Run-Side-001-Run-Side-004__SkateBoarding-Front-001-SkateBoarding-Front-004.gif?raw=true" width="16%">
  <img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Swing-Bench-001-Swing-Bench-004__Swing-Bench-005-Swing-SideAngle-002.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_Swing-SideAngle-003-Walk-Front-002__Walk-Front-003-Walk-Front-005.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V19_Singing1_snip_000016-V20_Singing2_snip_000019__V20_Singing2_snip_000020-V20_Singing2_snip_000023.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V21_Tap1_snip_000032-V26_Piano2_snip_000035__V26_Piano2_snip_000036-V28_Dog1_snip_000039.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V40_Guitar5_snip_000056-V40_Guitar5_snip_000059__V42_Violin2_snip_000060-V42_Violin2_snip_000063.gif?raw=true" width="16%"><img src="https://github.com/wz-zhao/Attend-to-Anything/blob/main/Fig/pred_overlay_pair_V7_Basketball2_snip_000088-V7_Basketball2_snip_000091__V7_Basketball2_snip_000092-V7_Basketball2_snip_000092.gif?raw=true" width="16%">
</div>


## 🌟 The first Unified Foundation Model for Attention Modeling (Saliency)


-  Human attention (saliency prediction) selectively focuses on informative regions when perceiving images, videos, and audio-visual scenes. However, **existing attention and saliency models are often tailored to specific scenarios, such as static images, videos, or isolated audio-visual setting**s, limiting their generalization in real-world applications.

-  We propose the **Attend to Anything Model (AAM)**, a unified foundation model for attention understanding across images, videos, and audio-visual scenes. AAM formulates attention as a **cognitive entailment relationship**, enabling prompt-driven reasoning over both salient regions and hierarchical task relationships. To connect static and dynamic attention, AAM further models video attention evolution as a temporally diffusive process inspired by fluid dynamics.

-  Experiments on **16 public benchmarks** show that AAM outperforms state-of-the-art methods by about **6%** on diverse attention and saliency tasks, while achieving a **4× speedup** in video inference. These results demonstrate AAM’s generality and efficiency as a foundation model for human attention prediction.

---
<!-- <p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_intro_1.png" width="70%">
</p>

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_sns_1.png" width="70%">
</p> -->

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

---

## 📂 Datasets

| Task | Training Datasets | Testing Datasets | Download |
|------|-------------------|------------------|----------|

---

## ✨  Visual Results


---

## ⚙️ Environment Setup

## 📚 Citation

If you find this repository useful, please use the following BibTeX entry for citation and give us a star⭐.

```bibtex


