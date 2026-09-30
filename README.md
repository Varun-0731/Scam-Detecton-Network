# GNN-Based Coordinated Scam Network Detection

A complete fraud detection pipeline that transforms financial transactions into a graph neural network–based investigator tool, using the PaySim synthetic mobile-money dataset.

## Architecture

```
PaySim CSV → Preprocessing → Feature Engineering → Graph Construction
                                                         ↓
        Streamlit Dashboard ← Community Detection ← GNN Training ← Baseline ML
```

## Quick Start

```bash
# 1. Activate virtual environment
cd scam-network-detection
source venv/bin/activate

# 2. Run full pipeline (10% sample for quick testing)
python run_pipeline.py --phase all --sample-frac 0.1

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

## Pipeline Phases

| Phase | Description | Command |
|-------|-------------|---------|
| 1 | Data Preprocessing | `python run_pipeline.py --phase 1` |
| 2 | Feature Engineering | `python run_pipeline.py --phase 2` |
| 3 | Graph Construction | `python run_pipeline.py --phase 3` |
| 4 | Baseline ML (LR, RF, XGBoost) | `python run_pipeline.py --phase 4` |
| 5 | GNN Models (GraphSAGE, GAT) | `python run_pipeline.py --phase 5` |
| 6 | Evaluation & Comparison | `python run_pipeline.py --phase 6` |
| 7 | Community Detection | `python run_pipeline.py --phase 7` |
| 8 | Summary & Dashboard | `python run_pipeline.py --phase 8` |

## Project Structure

```
scam-network-detection/
├── data/                          # Dataset files
│   └── PS_*.csv                   # PaySim dataset (471 MB)
├── src/                           # Source code
│   ├── data_preprocessing.py      # Load, clean, encode
│   ├── feature_engineering.py     # Account & edge features
│   ├── graph_construction.py      # PyTorch Geometric graph
│   ├── baseline_model.py          # LR, RF, XGBoost
│   ├── gnn_model.py               # GraphSAGE, GAT, FocalLoss
│   ├── evaluation.py              # Metrics, plots, comparison
│   └── community_detection.py     # Louvain, money-flow, risk
├── dashboard/                     # Streamlit dashboard
│   ├── app.py                     # Main entry point
│   ├── pages/                     # Dashboard pages
│   │   ├── 1_overview.py          # Dataset exploration
│   │   ├── 2_network_explorer.py  # Interactive graph viz
│   │   ├── 3_risk_analysis.py     # Model comparison
│   │   └── 4_community_detection.py # Suspicious networks
│   └── assets/style.css           # Premium dark theme
├── models/                        # Saved model weights
├── outputs/                       # Results & plots
├── run_pipeline.py                # End-to-end pipeline runner
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Dataset

**PaySim**: 6.36M synthetic mobile-money transactions
- 5 transaction types: PAYMENT, CASH_OUT, CASH_IN, TRANSFER, DEBIT
- 0.13% fraud rate (8,213 fraudulent transactions)
- Fraud occurs only in TRANSFER and CASH_OUT types

## Models

| Model | Type | Uses Graph? |
|-------|------|-------------|
| Logistic Regression | Baseline | No |
| Random Forest | Baseline | No |
| XGBoost | Baseline | No |
| GraphSAGE | GNN | Yes |
| GAT | GNN | Yes |

## Key Features

- **Account-level features**: 20+ engineered features per account
- **Graph-aware detection**: GNN leverages neighborhood structure
- **Focal Loss**: Handles extreme class imbalance (0.13% fraud)
- **Community detection**: Louvain algorithm identifies coordinated fraud rings
- **Money-flow tracing**: BFS-based fund movement analysis
- **Interactive dashboard**: Investigator-facing Streamlit UI

## Tech Stack

Python 3.13 · PyTorch · PyTorch Geometric · scikit-learn · XGBoost · NetworkX · Pandas · Plotly · Streamlit
