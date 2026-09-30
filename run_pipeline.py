#!/usr/bin/env python3
"""GNN-Based Coordinated Scam Network Detection - Full Pipeline Runner

Usage:
    python run_pipeline.py --phase all --sample-frac 0.1
    python run_pipeline.py --phase 1      # Run only data preprocessing
    python run_pipeline.py --phase 4      # Run only baseline models
"""

import argparse
import time
import os
import sys
import importlib
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_phase(phase_num, name, func, kwargs=None):
    """Run a specific phase function and measure time."""
    print(f"\n{'='*60}")
    print(f"  Phase {phase_num}: {name}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = func(**(kwargs or {}))
        elapsed = time.time() - start_time
        print(f"\n✅ Phase {phase_num} completed in {elapsed:.1f}s")
        return True, result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Phase {phase_num} failed after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False, None


def phase1_preprocessing(data_path, sample_frac):
    """Phase 1: Data Preprocessing"""
    from src.data_preprocessing import load_data, clean_data, encode_data, get_fraud_summary

    df = load_data(data_path, sample_frac=sample_frac)
    df = clean_data(df)
    df = encode_data(df)

    summary = get_fraud_summary(df)
    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(PROJECT_ROOT, 'data', 'processed_transactions.pkl')
    df.to_pickle(out_path)
    print(f"\nSaved processed data to {out_path}")
    return df


def phase2_features():
    """Phase 2: Feature Engineering"""
    from src.feature_engineering import (
        compute_account_features, label_accounts,
        compute_edge_features, scale_features
    )
    import pandas as pd

    df = pd.read_pickle(os.path.join(PROJECT_ROOT, 'data', 'processed_transactions.pkl'))

    print("Computing account features...")
    account_features = compute_account_features(df)

    print("Labeling accounts...")
    account_features = label_accounts(df, account_features)

    print("Computing edge features...")
    edge_features = compute_edge_features(df)

    # Determine numeric feature columns for scaling
    exclude_cols = {'account_id', 'is_fraud'}
    feature_cols = [c for c in account_features.select_dtypes(include='number').columns
                    if c not in exclude_cols]

    print("Scaling features...")
    account_features, scaler = scale_features(account_features, feature_cols)

    account_features.to_pickle(os.path.join(PROJECT_ROOT, 'data', 'account_features.pkl'))
    edge_features.to_pickle(os.path.join(PROJECT_ROOT, 'data', 'edge_features.pkl'))
    print(f"Saved account_features ({len(account_features)} accounts) and edge_features ({len(edge_features)} edges)")
    return account_features, edge_features


def phase3_graph():
    """Phase 3: Graph Construction"""
    import pandas as pd
    from src.graph_construction import build_pyg_graph, save_graph

    account_features = pd.read_pickle(os.path.join(PROJECT_ROOT, 'data', 'account_features.pkl'))
    edge_features = pd.read_pickle(os.path.join(PROJECT_ROOT, 'data', 'edge_features.pkl'))

    exclude_cols = {'account_id', 'is_fraud'}
    feature_cols = [c for c in account_features.select_dtypes(include='number').columns
                    if c not in exclude_cols]

    graph_data = build_pyg_graph(account_features, edge_features, feature_cols)
    save_graph(graph_data, os.path.join(PROJECT_ROOT, 'data', 'graph_data.pt'))
    return graph_data


def phase4_baselines():
    """Phase 4: Baseline ML Models"""
    from src.baseline_model import run_baselines
    import json

    results = run_baselines(os.path.join(PROJECT_ROOT, 'data', 'account_features.pkl'))

    # Save results
    serializable = {}
    for name, metrics in results.items():
        serializable[name] = {k: v for k, v in metrics.items()}

    with open(os.path.join(PROJECT_ROOT, 'outputs', 'baseline_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print("Saved baseline results to outputs/baseline_results.json")
    return results


def phase5_gnn():
    """Phase 5: GNN Models"""
    from src.gnn_model import run_gnn_pipeline
    import json

    results = run_gnn_pipeline(os.path.join(PROJECT_ROOT, 'data', 'graph_data.pt'))

    serializable = {}
    for name, metrics in results.items():
        serializable[name] = {k: v for k, v in metrics.items()}

    with open(os.path.join(PROJECT_ROOT, 'outputs', 'gnn_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print("Saved GNN results to outputs/gnn_results.json")
    return results


def phase6_evaluation():
    """Phase 6: Evaluation & Comparison"""
    import json

    baseline_path = os.path.join(PROJECT_ROOT, 'outputs', 'baseline_results.json')
    gnn_path = os.path.join(PROJECT_ROOT, 'outputs', 'gnn_results.json')

    all_results = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            all_results.update(json.load(f))
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            all_results.update(json.load(f))

    if not all_results:
        print("No results found. Run phases 4 and 5 first.")
        return

    from src.evaluation import print_comparison_table
    comparison_df = print_comparison_table(all_results)
    comparison_df.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'model_comparison.csv'))
    print("Saved comparison to outputs/model_comparison.csv")
    return comparison_df


def phase7_community():
    """Phase 7: Community Detection"""
    from src.community_detection import run_community_detection

    summary = run_community_detection(
        account_features_path=os.path.join(PROJECT_ROOT, 'data', 'account_features.pkl'),
        edge_features_path=os.path.join(PROJECT_ROOT, 'data', 'edge_features.pkl'),
        embeddings_path=os.path.join(PROJECT_ROOT, 'data', 'node_embeddings.npy')
    )
    return summary


def phase8_summary():
    """Phase 8: Summary & Dashboard Instructions"""
    print("\n" + "="*60)
    print("  🎉  PIPELINE COMPLETE!")
    print("="*60)
    print("""
Generated files:
  data/processed_transactions.pkl  — Cleaned transaction data
  data/account_features.pkl        — Account-level features
  data/edge_features.pkl           — Edge-level features
  data/graph_data.pt               — PyTorch Geometric graph
  data/node_embeddings.npy         — GNN node embeddings
  outputs/baseline_results.json    — Baseline model metrics
  outputs/gnn_results.json         — GNN model metrics
  outputs/model_comparison.csv     — All models comparison
  models/                          — Saved model weights

To launch the interactive dashboard:
  cd {project_root}
  source venv/bin/activate
  streamlit run dashboard/app.py
""".format(project_root=PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Based Coordinated Scam Network Detection Pipeline"
    )
    parser.add_argument('--phase', type=str, default='all',
                        help="Phase to run: 1-8 or 'all' (default: all)")
    parser.add_argument('--data-path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'PS_20174392719_1491204439457_log.csv'),
                        help="Path to PaySim CSV")
    parser.add_argument('--sample-frac', type=float, default=0.1,
                        help="Fraction of legitimate data to sample (default: 0.1). "
                             "All fraud cases are always kept.")

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)

    phases = [
        (1, "Data Preprocessing",    lambda: phase1_preprocessing(args.data_path, args.sample_frac)),
        (2, "Feature Engineering",    phase2_features),
        (3, "Graph Construction",     phase3_graph),
        (4, "Baseline ML Models",     phase4_baselines),
        (5, "GNN Models",            phase5_gnn),
        (6, "Evaluation",            phase6_evaluation),
        (7, "Community Detection",    phase7_community),
        (8, "Summary & Dashboard",   phase8_summary),
    ]

    # Determine which phases to run
    if args.phase.lower() == 'all':
        target_phases = phases
    else:
        try:
            idx = int(args.phase)
            if 1 <= idx <= 8:
                target_phases = [phases[idx - 1]]
            else:
                print("Phase must be between 1 and 8")
                sys.exit(1)
        except ValueError:
            print("Invalid phase. Use 1-8 or 'all'")
            sys.exit(1)

    total_start = time.time()
    timings = []

    for phase_num, name, func in target_phases:
        phase_start = time.time()
        success, _ = run_phase(phase_num, name, func)
        elapsed = time.time() - phase_start
        timings.append((phase_num, name, elapsed, success))

        if not success and phase_num < 8:
            print(f"\n⚠️  Pipeline stopped at Phase {phase_num}. Fix the error and re-run.")
            break

    # Print timing summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline Timing Summary")
    print(f"{'='*60}")
    for num, name, elapsed, success in timings:
        status = "✅" if success else "❌"
        print(f"  {status} Phase {num}: {name:<25s} {elapsed:>8.1f}s")
    print(f"  {'─'*50}")
    print(f"  Total: {total_elapsed:>37.1f}s")


if __name__ == '__main__':
    main()
