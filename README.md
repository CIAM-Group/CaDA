# CaDA

The official code repository for [CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention](https://arxiv.org/abs/2412.00346).

## Preparation

1. **Clone the repository**:

```bash
git clone https://github.com/CIAM-Group/CaDA.git
```

2. **Download datasets**:

* Download `data.zip` from [Hugging Face](https://huggingface.co/datasets/Goodyee/CaDA/tree/main).
* Unzip `data.zip` and organize the files in the project directory as follows:

```
CaDA
├── data
├── 50
├── 100
└── utils
```

3. **Download checkpoints**:

* Download `checkpoint.zip` from [Hugging Face](https://huggingface.co/datasets/Goodyee/CaDA/tree/main).
* Unzip `checkpoint.zip`. It contains two folders named `50` and `100`. Move their contents into the project directory as shown:

```
CaDA
├── data
├── 50
│   └── result
│       └── (folders from checkpoint/50/)
├── 100
│   └── result
│       └── (folders from checkpoint/100/)
└── utils
```

## Training and Testing

For detailed instructions on training and testing the model, please refer to the README files inside the `50` and `100` directories.
