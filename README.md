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
│   ├── lib_data
│   └── synthetic_data
├── 50
├── 100
└── utils
```

3. **Download checkpoints**:

* Create 'result' folder manually under 'CaDA/50' and 'CaDA/100'.
* Download `checkpoint.zip` from [Hugging Face](https://huggingface.co/datasets/Goodyee/CaDA/tree/main).
* Unzip `checkpoint.zip`. It will produce two directories: `50` and `100`.
  * Inside `50`, you will find a folder named `2024-1111-1139`.
  * Inside `100`, you will find a folder named `2024-1121-1355`.
* Organize them into the project directory as follows:

```
CaDA
├── data
│   ├── lib_data
│   └── synthetic_data
├── 50
│   └── result
│       └── 2024-1111-1139
├── 100
│   └── result
│       └── 2024-1121-1355
└── utils
```

4. **Prepare environment**:

The project is developed with Python 3.8.15. Key packages include:
```
torch     2.0.1
torchrl   0.1.1
rl4co     0.2.0
tensordict                   0.1.2
```
The complete list of dependencies can be found in `requirements.txt`.


## Training and Testing

For detailed instructions on training and testing the model, please refer to the README files inside the `50` and `100` directories.
