# Physics-Informed Graph Neural Networks for Particle Momentum Estimation in CMS Trigger System


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
git clone https://github.com/[username]/gnn-particle-momentum-cms-trigger.git
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

[Graph Dataset Creation](https://github.com/[username]/gnn-particle-momentum-cms-trigger/blob/main/Graph_creation)

- [x] **Each station as a node :** 
- [x] **Each feature as a node :**
- [x] **BendingAngle as a node :**
- [x] **Eta Value as a node :**


### 1. Each Station as a Node

In this approach, each of the four stations is treated as a separate node in the graph. The node features correspond to the 7 features received at each respective station.

- **Total nodes**: 4 (one per station)  
- **Node feature length**: 7  
- **Edges**: To be determined — we are still exploring the optimal way to define edges between the four station nodes.

---

### 2. Each Feature as a Node

In this approach, each of the 7 features (across all stations) is treated as a node. The node features are the values of that particular feature across the 4 different stations.

- **Total nodes**: 7 (one per feature)  
- **Node feature length**: 4  
- **Edges**: To be determined — the best strategy for connecting these 7 feature nodes is still under investigation.

---

### 3. Bending Angle as a Node

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

### 4. Eta Value as a Node

In this final approach:

- **Node feature**: `η (eta)`  
- **Edge features**: `sin(ϕ)`, `cos(ϕ)`, and bending angle  
- **Graph type**: Fully connected graph

This setup is derived from our earlier observations, emphasizing η as a more informative representation of direction than θ, and leveraging meaningful edge features from angular and bending information.



















