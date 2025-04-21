# Graph Neural Networks for Particle Momentum Estimation in the CMS Trigger System

## Previous Work

This project builds upon prior efforts and research in identical direction:

- **Shravan Chaudhari**
  - Utilized the Boosted Top dataset from [this paper](https://arxiv.org/abs/2104.14659)
  - Repository: [GSoC2021_BoostedTopJets](https://github.com/Shra1-25/GSoC2021_BoostedTopJets)

- **Emre Kurtoglu**
  - [Project blog post](https://medium.com/@emre.kurt.96/gsoc-2021-graph-neural-networks-for-particle-momentum-estimation-in-the-cms-trigger-system-2216e4e4d005)

---

## Project Overview

Momentum is a key quantity in physics, defined as the product of mass and velocity, and is crucial for understanding particle interactions. The **muon** is a heavy lepton (~207× the mass of an electron), and plays a central role in high-energy physics experiments like those at the **Compact Muon Solenoid (CMS)** detector in the **Large Hadron Collider (LHC)**.

CMS identifies particles from collision events, and analyzing the momentum of **high-energy muons** is essential for reducing background noise and improving trigger accuracy, especially for the **Endcap Muon Track Finder (EMTF)**. As discussed in [this paper](https://iopscience.iop.org/article/10.1088/1742-6596/1085/4/042042), distinguishing between signal and background muons is vital for efficient event selection.

### Input Features

A muon passing through the endcap leaves hits in up to four stations (1–4). Each station provides 7 features:

- Phi
- Theta
- Bending angle
- Time
- Ring number
- Front/rear hit indicator
- Mask

Additionally, 3 road-level features are included:

- Pattern straightness
- Zone
- Median theta

Total features per event: `7 × 4 + 3 = 31`

### Goal

Estimate the **momentum** of muons using various learning models. Previously, models like BDTs, FCNNs, and CNNs were applied. In this repository, we explore **Graph Neural Networks (GNNs)** for more structured and physics-aligned modeling.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger.git
cd ./gnn-particle-momentum-cms-trigger/modular_code
```

### 2. Create a Virtual Environment

```bash
conda create -n gnn-particle python=3.8
conda activate gnn-particle
pip install -r requirements.txt
```

---

## Usage

### Using Bending Angle as Node Feature

```bash
python main.py --input_csv "/path/to/CMS_trigger.csv" \
               --batch_size 32 \
               --learning_rate 0.001 \
               --node_feat "bendAngle" \
               --epochs 18 \
               --save_dir "./outputs"
```

### Using Eta Value as Node Feature

```bash
python main.py --input_csv "/path/to/CMS_trigger.csv" \
               --batch_size 32 \
               --learning_rate 0.001 \
               --node_feat "etaValue" \
               --epochs 18 \
               --save_dir "./outputs"
```

---

## Datasets

The dataset contains **1,179,356 muon events** generated using **Pythia**, available on [Kaggle](https://www.kaggle.com/datasets/ekurtoglu/cms-dataset).


## Graph Construction Strategies

[Graph Dataset Creation](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Graph_creation)

- [x] **Each station as a node :** 
- [x] **Each feature as a node :**
- [x] **BendingAngle as a node :**
- [x] **Eta Value as a node :**


### 1. [Each Station as a Node](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Graph_creation/Converting_to_graphs_eachstation_node.ipynb)

In this approach, each of the four stations is treated as a separate node in the graph. The node features correspond to the 7 features received at each respective station.

- **Total nodes**: 4 (one per station)  
- **Node feature length**: 7  
- **Edges**: To be determined — we are still exploring the optimal way to define edges between the four station nodes.

---

### 2. [Each Feature as a Node](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Graph_creation/Converting_to_graphs_eachfeature_node.ipynb)

In this approach, each of the 7 features (across all stations) is treated as a node. The node features are the values of that particular feature across the 4 different stations.

- **Total nodes**: 7 (one per feature)  
- **Node feature length**: 4  
- **Edges**: To be determined — the best strategy for connecting these 7 feature nodes is still under investigation.

---

### 3. [Bending Angle as a Node](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/bendingAngle_node/A1/readme.md)

Although the above two methods produced good results in terms of MSE/MAE, we found that the number of parameters in the GNN was nearly 10× higher than our baseline model (TabNet).

This led us to perform extensive feature engineering, and we made the following observations:

- **Dropped Features**:  
  Mask values, frontValue, time info, ring number, and front/rear hit features were nearly constant across particles and contributed very little to momentum prediction.  
- **Phi Angle**:  
  Using `sin(ϕ)` and `cos(ϕ)` instead of raw ϕ yielded better performance.  
- **Theta Angle**:  
  Replaced with `η (eta)` value, which was more meaningful in this context.  

After refining our feature set, we constructed a new graph:

- **Node feature**: Bending angle  
- **Edge features**: `sin(ϕ)`, `cos(ϕ)`, and `η`  
- **Graph type**: Fully connected graph

![image](https://github.com/user-attachments/assets/84e22d1b-254e-4ac1-a986-0225deb06888)

---

### 4. [Eta Value as a Node](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A2/readme.md)

In this final approach:

- **Node feature**: `η (eta)`  
- **Edge features**: `sin(ϕ)`, `cos(ϕ)`, and bending angle  
- **Graph type**: Fully connected graph

This setup is derived from our earlier observations, emphasizing η as a more informative representation of direction than θ, and leveraging meaningful edge features from angular and bending information.

---

## Experiment Results for 1 and 2

| Expt_NO | approach_of_graph  | Edges                    | Model                | Loss_fnc | loss | MAE  | Accuracy | F1 Score | Notebook |
|---------|---------------------|-----------------------|----------------------|----------|------|------|----------|----------|----------|
| 1       | each_station        | 0-1-2-3                  | 4 GCN                | pT Loss  | 3.83 | 43.21| 80.18    | 0.167    | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v2_A1/eachstation_node_v2.ipynb)     |
| 2       | each_station        | fully connected          | 4 MPL(embed-128)     | pT Loss  | 1.364| 15.82| 96.02    | 0.5529   | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v3_A1/eachstation_node_v3.ipynb)     |
| 3       | each_station        | fully connected          | 4 MPL(embed-64)      | pT Loss  | 1.3384|	16.22|	95.8146|	0.5422  | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v4_A1/eachstation_node_v4.ipynb)     |
| 4       | each_feature        | 0-1-2-3 // 2- (0,4, 5, 6)| 4 GCN                | pT Loss  | 4.38 | 45.33| 80.81    | 0.14     | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachfeature_node/v1_A2/eachfeature_node_v1.ipynb)     |
| 5       | each_feature        | fully connected          | 4 MPL                | pT Loss  | 3.7945|	41.7085|	80.337|	0.1682    | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachfeature_node/v2_A2/eachfeature_node_v2.ipynb)     |
| 6       | each_station        | fully connected          | 4 MPL(embed-64)      | MSE Loss  |0.000478|	52.64|	62.1498|	0.16685  | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v6_A1/Eachstation_node.ipynb)     |
| 7       | each_station        | fully connected          | 4 MPL(embed-64)      | MSE Loss  | 0.00263|	13.529|	97.458|	0.0515  | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v7_A1/Eachstation_node.ipynb)     |
| 8       | each_station        | 0-1-2-3                  | 4 MPL(embed-64)      | MSE Loss  | 0.00191|	13.3|	97.4053|	0.000326  | [link]()     |
| 9       | each_station        | fully connected          | 4 MPL(embed-64)      | pT Loss  | 0.004164|	2.0716|	100|	0.0  | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v9_A1/Eachstation_node.ipynb)     |
| 11       | each_station        | fully connected          | 4 MPL(embed-64)      | Custom Loss  | 0.46798|	0.8178|	100|	0.0  | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v11_A1/Eachstation_node_v11.ipynb)     |
| 12       | each_station        | fully connected          | 4 MPL(embed-64)      | Custom Loss  | 0.4538|	0.7849|	100|	0.0  | [link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/Eachstation_node/v12_A1/eachstation_node_v12.ipynb)     |


## Experiment Results for 3 and 4

| **Model**            | **MAE**   | **MSE**   | **Avg Inference Time (μs)** | **No. of Parameters** | **Model Code Link**         |
|----------------------|-----------|-----------|------------------------------|-----------------------|------------------------------|
| **TabNet**           | 0.9607    | 2.9746    | 458.7                        | 6696                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/baseline_tabnet/tabnet.ipynb)  |
| GNN-bendAngle    | 1.202931  | 3.520059  | 522.204                      | 5579                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/bendingAngle_node/A1/bendingAngle_node.ipynb)  |
| GNN-bendAngle2   | 1.215189  | 3.605358  | 530.19                            | 5903                  | [Link](url)  |
| GNN-etaValue1    | 1.146910  | 3.240220  | 522.204                      | 5579                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A1/eta_value_1.ipynb)  |
| **GNN-etaValue2**    | 0.992087  | 2.525927  | 530.19                            | 5903                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A2/eta_value_2.ipynb)  |
| GNN-etaValue3    | 1.145697         | 3.276628         | 614.565                            | 6437                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A3/eta_value_3.ipynb)  |
| GNN-etaValue4    | 1.133285         | 3.205457         | 267.509                             | 6112                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A4/eta_value_4.ipynb)  |
| GNN-etaValue5    | 0.941620  | 2.312492  | 515.472                      | 6545                  | [Link](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger/blob/main/Models/etaValue_node/A5/eta_value_5.ipynb)  |

We observed that the GNN model, when using the **eta (η) value as the sole node feature** and **sin(ϕ), cos(ϕ), and bending angle as edge features**, outperformed the TabNet baseline in terms of both **MAE** and **MSE**, despite having a comparable number of parameters. For a detailed breakdown of each model, please refer to the corresponding [code repository](https://github.com/Abrar2652/gnn-particle-momentum-cms-trigger).

This demonstrates that **GNNs are well-suited for extracting meaningful representations from muon detector data** and can accurately estimate particle momentum by leveraging spatial and angular relationships.

















