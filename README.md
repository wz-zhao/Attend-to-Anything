#  👀 Attend to Anything Model (AAM) 
> **The official implementation of [ICML 2026] [Attend to Anything: Foundation Model for Unified Human Attention Modeling](https://icml.cc/virtual/2026/poster/61103)**

<div align="center">
  
[![Paper](https://img.shields.io/badge/📜_Paper-ArXiv-red)](https://arxiv.org/abs/2606.03540)
[![License](https://img.shields.io/badge/📄_License-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/🐍_Python-3.10+-blue)]()
[![CN-Version](https://img.shields.io/badge/🇨🇳_Paper-README_CN-blue)](Paper_CN.md)

</div>



<!-- <p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_show_1.png" width="90%"> -->
---




## 🌟 The first Unified Foundation Model for Attention Modeling (Saliency Prediction)


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


