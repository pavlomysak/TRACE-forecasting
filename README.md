# TRACE: Temporal Recurrent Attention Convolutional Encoder  
*A hybrid deep learning architecture for intermittent demand forecasting*

---

## 🌎 Overview

**TRACE** (Temporal Recurrent Attention Convolutional Encoder) is a deep learning architecture designed to address the **intermittent time series forecasting problem**, where traditional models often fail due to sparse, bursty, and highly non-stationary demand patterns.

Developed by **Pavlo Mysak** during his work as an *eCommerce Analyst* at **La Tourangelle**, TRACE combines the strengths of **feed-forward**, **convolutional**, **recurrent**, and **attention-based** architectures into a unified encoder for robust temporal representation learning.

This model was built from scratch in **PyTorch** to forecast future sales and demand across thousands of SKUs, balancing both **classification (demand occurrence)** and **regression (demand magnitude)** objectives.

---

## 🧠 Motivation

Intermittent forecasting presents unique challenges:
- Long stretches of zero demand
- Abrupt bursts or seasonal spikes
- Heterogeneous covariate effects (time, category, SKU, etc.)

Standard RNNs or temporal CNNs often underperform under these dynamics.  
TRACE was designed to **combine local pattern extraction, recurrent memory, and global temporal attention** — allowing it to capture both short- and long-range dependencies in sparse sequences.

---

## ⚙️ Architecture

The TRACE model builds a hierarchical encoder composed of multiple **FCR (Feedforward–Convolution–Recurrent)** blocks, followed by **multi-head self-attention** and **additive attention pooling**.

### 🔸 1. Input Structure
TRACE operates on multivariate time series consisting of:
- **Historical target variable** (e.g., past sales or demand)
- **Numeric covariates** (e.g., prices, promotions, seasonality features)
- **Categorical covariates** (e.g., SKU, product family, subcategory)
- **Datetime embeddings** (month, week, day)

Each component is embedded or normalized and concatenated into a unified temporal tensor.

### 🔸 2. FCR Block
Each FCR block processes temporal information through three layers of abstraction:
1. **Feedforward Expansion** – nonlinear transformation and residual connection  
2. **Depthwise–Pointwise Convolution** – local temporal pattern extraction  
3. **GRU Layer** – recurrent modeling of sequential dependencies  

Residual connections and dropout are applied throughout to stabilize learning.

### 🔸 3. Attention Encoding
After the FCR stack, TRACE applies **multi-head self-attention** to capture global dependencies across time.  
This is followed by **additive attention pooling**, producing a context vector summarizing the sequence through learned temporal importance weights.

### 🔸 4. Dual Output Heads
TRACE supports two simultaneous prediction heads:
- **Classification Head:** Predicts probability of future demand occurrence  
- **Regression Head:** Predicts the magnitude of demand when it occurs  

Both heads use a **maxout**-style nonlinear transformation before projection to the final forecast horizon.

---

## 📈 Key Advantages

| **Component** | **Purpose** | **Benefit** |
|----------------|-------------|-------------|
| Feedforward Layer | Nonlinear feature expansion | Captures high-order interactions |
| Depthwise–Pointwise Convolution | Local temporal filters | Efficient pattern recognition |
| GRU Layer | Sequential memory | Handles temporal persistence |
| Multi-Head Attention | Global context modeling | Captures long-range dependencies |
| Additive Attention Pooling | Context summarization | Robust sequence-to-vector encoding |
| Dual Heads | Classification + Regression | Suited for intermittent demand forecasting |

---

## 📊 Applications

- 🛒 Retail & eCommerce demand forecasting  
- 📦 Inventory optimization for low-volume SKUs  
- 💰 Price and promotion impact modeling  
- 📈 Hybrid sales classification–regression tasks  
- ⏳ Sparse and irregular time series prediction  

---

## ⚙️ Technical Details

- **Framework:** PyTorch (v2.0+)  
- **Language:** Python 3.10+  
- **Core Layers:** LayerNorm, Conv1D (Depthwise + Pointwise), GRU, Multi-Head Attention  
- **Loss Functions:**  
  - Binary Cross-Entropy (classification head)  
  - Mean Squared Error (regression head)  
- **Optimizer:** Adam / AdamW  
- **Learning Rate Scheduler:** CosineAnnealingLR or ReduceLROnPlateau  
- **Initialization:** Xavier Uniform  
- **Training Context:**  
  Originally developed and trained on **SKU-level weekly data** from *La Tourangelle’s eCommerce operations*, incorporating structured covariates and calendar features to handle highly intermittent demand.

