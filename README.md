# 🔍 Protecting Your Video Content: Disrupting Automated Video-based LLM Annotations (CVPR 2025)

  
📄 Accepted to **CVPR 2025**  
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b.svg)](https://arxiv.org/abs/2503.21824)   

---

## 📝 Abstract

Recently, video-based large language models (video-based LLMs) have achieved impressive performance across various video comprehension tasks. However, this rapid advancement raises significant privacy and security concerns, particularly regarding the unauthorized use of personal video data in automated annotation by video-based LLMs. These unauthorized annotated video-text pairs can then be used to improve the performance of downstream tasks, such as text-to-video generation. To safeguard personal videos from unauthorized use, we propose two series of protective video watermarks with imperceptible adversarial perturbations, named Ramblings and Mutes. Concretely, Ramblings aim to mislead video-based LLMs into generating inaccurate captions for the videos, thereby degrading the quality of video annotations through inconsistencies between video content and captions. Mutes, on the other hand, are designed to prompt video-based LLMs to produce exceptionally brief captions, lacking descriptive detail. Extensive experiments demonstrate that our video watermarking methods effectively protect video data by significantly reducing video annotation performance across various video-based LLMs, showcasing both stealthiness and robustness in protecting personal video content.


---

## 📦 Installation

```bash
git clone https://github.com/ttthhl/Protecting_Your_Video_Content.git
```
Then you can follow [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) to configure the environment.

---

## Data Preparation

You can download the dataset from [nkp37/OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M).
And init the csv file in the following format.
```bash
video_name
...
...
```

---
## Model Preparation
You can download the model from [DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained). And configure the llama_model and ckpt in video_llama_eval_only_vl.yaml of eval_configs following [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA).

---

### Running code
Set the gt_file(the csv path), video_dir(video's folder path), output_dir in attack.sh.
```bash
bash attack.sh
```

random is Rambling-F. train is Rambling-L. eos is Mute-N. eos2 is Mute-S. 

### Citation
```bash
@inproceedings{liu2025protecting,
  title={Protecting your video content: Disrupting automated video-based llm annotations},
  author={Liu, Haitong and Gao, Kuofeng and Bai, Yang and Li, Jinmin and Shan, Jinxiao and Dai, Tao and Xia, Shu-Tao},
  booktitle={CVPR},
  year={2025}
}
```
### Acknowledgements
This respository is mainly based on [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA). Thanks for their wonderful works!