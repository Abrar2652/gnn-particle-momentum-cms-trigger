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

