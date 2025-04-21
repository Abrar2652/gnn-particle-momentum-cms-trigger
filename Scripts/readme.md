
# Graph Neural Network for Particle Momentum Prediction

## 🧠 Overview

This project focuses on training and evaluating a **Graph Neural Network (GNN)** to predict particle momentum in a high-energy physics context. It provides scripts for data preparation, model training and evaluation, and visualization of results.

### 🔍 Key Observations

1. **Node Feature Selection**:
   - Initially, node features were derived either per station or per feature.
   - We found that using only the **bending angle** per station performed reasonably well.
   - However, replacing the bending angle with **η (eta)** values as node features yielded significantly better results.

2. **Edge Feature Design**:
   - Introduced a 3-dimensional edge feature vector composed of:
     - `sin(φ)`
     - `cos(φ)`
     - `η = -log(tan(θ / 2))`

3. **Model Architecture**:
   - We used these node and edge features to design a message-passing GNN.
   - Several configurations with varying hidden layers were evaluated.
   - Results and comparisons are presented in the Results section.

---

## 📁 Project Structure

| File          | Description |
|---------------|-------------|
| `dataset.py`  | Data loading and preprocessing logic. |
| `losses.py`   | Custom loss functions for training. |
| `main.py`     | Main script for training, testing, and plotting results. |
| `models.py`   | Definition of GNN model architectures. |
| `train.py`    | Dedicated script for model training. |
| `utils.py`    | Helper functions for evaluation and visualization. |
| `requirements.txt` | Python dependencies for the project. |
| `readme.md`   | Project documentation (this file). |

---

## ⚙️ Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger.git
   cd ./gnn-particle-momentum-cms-trigger/modular_code
   ```

2. **Create a Virtual Environment**:
   ```bash
   conda create -n gnn-particle python=3.10
   conda activate gnn-particle
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Using Bending Angle as Node Feature
```bash
python main.py --input_csv "/path/to/CMS_trigger.csv" \
               --batch_size 32 \
               --learning_rate 0.001 \
               --node_feat "bendAngle" \
               --epochs 18 \
               --save_dir "./outputs"
```

### 2. Using Eta Value as Node Feature
```bash
python main.py --input_csv "/path/to/CMS_trigger.csv" \
               --batch_size 32 \
               --learning_rate 0.001 \
               --node_feat "etaValue" \
               --epochs 18 \
               --save_dir "./outputs"
```

---

## 📊 Experiment Results

| **Model**         | **MAE**   | **MSE**   | **Avg Inference Time (μs)** | **# Parameters** | **Notebook Link** |
|------------------|-----------|-----------|------------------------------|------------------|-------------------|
| **TabNet**        | 0.9607    | 2.9746    | 458.7                        | 6696             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/baseline_tabnet/tabnet.ipynb) |
| GNN-bendAngle     | 1.2029    | 3.5201    | 522.2                        | 5579             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/bendingAngle_node/A1/bendingAngle_node.ipynb) |
| GNN-bendAngle2    | 1.2152    | 3.6054    | 530.2                        | 5903             | [View](url) |
| GNN-etaValue      | 1.1469    | 3.2402    | 522.2                        | 5579             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A1/eta_value_1.ipynb) |
| **GNN-etaValue2** | **0.9921**| **2.5259**| 530.2                        | 5903             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A2/eta_value_2.ipynb) |
| GNN-etaValue3     | 1.1457    | 3.2766    | 614.6                        | 6437             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A3/eta_value_3.ipynb) |
| GNN-etaValue4     | 1.1333    | 3.2055    | **267.5**                    | 6112             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A4/eta_value_4.ipynb) |
| GNN-etaValue5     | **0.9416**| **2.3125**| 515.5                        | 6545             | [View](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A5/eta_value_5.ipynb) |

> 📌 **Note:** GNN models with `etaValue` as node features consistently outperformed those using `bendAngle`, both in terms of error metrics and generalization.

---

## 📬 Contributing

We welcome contributions! Please open issues or submit pull requests to suggest improvements, fix bugs, or add new features.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
