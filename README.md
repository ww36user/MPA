# MPA
This is the code for our paper "MPA: Multimodal Prototype Augmentation for Few-Shot Learning" that has been accepted by AAAI-2026!

# Requirements

We adopted Python version 3.10.14.

```
pip install -r requirements.txt
```



# Datesets
You can download these datasets from this (address: https://pan.baidu.com/s/1aJvPyprYEWhHiH7TGHaNvw,
password: MPA6).

# Test for EuroSAT:
```
- 5-way 1-shot:
CUDA_VISIBLE_DEVICES=0 python test.py --n_support 1 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 100;

- 5-way 5-shot:
CUDA_VISIBLE_DEVICES=0 python test.py --n_support 5 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 100;

```

# Thanks
Thanks to these works for their contributions: [CLIP](https://github.com/openai/CLIP) for providing the pretrained models, and [LDP-Net](https://github.com/NWPUZhoufei/LDP-Net) for offering the code framework.

# Any questions:
If you have any questions, please open an issue on GitHub or contact me via email at **weiwang@stu.ynu.edu.cn**.
