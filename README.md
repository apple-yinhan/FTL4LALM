# FTL4LALM

## Abstract

Large audio language models (LALMs) are a class of foundation models for audio understanding. Existing LALMs tend to degrade significantly in real-world noisy acoustic conditions where speech and non-speech sounds interfere. While noise-aware fine-tuning can improve robustness, it requires task-specific noisy data and expensive retraining, limiting scalability. To address this issue, we propose Focus-Then-Listen (FTL), a plug-and-play audio enhancer that improves LALMs' noise robustness. Specifically, FTL first separates the input waveform into speech and non-speech, and a modality router is applied to predict the target audio modality (e.g., speech) based on the user's instruction. Finally, a modality-aware fusion block generates a task-adaptive enhanced signal for improved downstream perception and reasoning. Experiments across multiple LALMs and tasks show that FTL improves performance across different noise levels without fine-tuning on LALMs.

**Paper Link**: TBD


## Running FTL

1. Prepare Environment

conda env create -f environment.yml

2. Run the code

Please firstly download the checkpoint of the separator (SNSep) from huggingface: https://huggingface.co/apple121/FTL4LALM

Python ./ftl.py

## MMAU-Pro-Ctrl

You can download MMAU-Pro-Ctrl from huggingface: https://huggingface.co/datasets/apple121/MMAU-Pro-Ctrl 

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yin2026focusthen,
  title     = {Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models},
  author    = {Han Yin and Yang Xiao and Younghoo Kwon and Ting Dang and Jung-Woo Choi},
  booktitle = {ICML 2026 Workshop on Machine Learning for Audio (Learning to Listen)},
  year      = {2026},
  url       = {https://mlforaudioworkshop.github.io/}
}
```

