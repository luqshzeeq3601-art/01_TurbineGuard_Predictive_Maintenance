"""Command-line interface for TurbineGuard."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from turbineguard.config import load_config
from turbineguard.data import (
    load_raw_cmapss_file,
    load_raw_rul_file,
    run_data_quality_checks,
    save_quality_report,
)
from turbineguard.features import (
    extract_causal_features_for_dataframe,
    extract_snapshot_features,
)
from turbineguard.labels import assign_anomaly_proxies, compute_rul_labels
from turbineguard.splits import (
    create_engine_splits,
    create_snapshot_manifest,
    save_snapshot_manifest,
    save_split_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_prepare(config_path: str | Path) -> None:
    """
    Execute data preparation pipeline:
    1. Validate raw FD001 data and generate data quality report.
    2. Create disjoint 80/20 engine split and 5-fold grouped CV assignments.
    3. Create frozen validation snapshot manifest.
    4. Compute causal rolling features for dev, val, and test partitions.
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_file = raw_dir / "train_FD001.txt"
    test_file = raw_dir / "test_FD001.txt"
    rul_file = raw_dir / "RUL_FD001.txt"

    if not (train_file.exists() and test_file.exists() and rul_file.exists()):
        raise FileNotFoundError(f"Missing raw data files in {raw_dir}. Run download_data.py first.")

    logger.info("Loading raw data files...")
    train_df = load_raw_cmapss_file(train_file, "train")
    test_df = load_raw_cmapss_file(test_file, "test")
    rul_array = load_raw_rul_file(rul_file)

    logger.info("Running data quality checks...")
    quality_rep = run_data_quality_checks(train_df, test_df, rul_array)
    save_quality_report(quality_rep, Path(cfg.data.quality_report))
    logger.info(f"Saved data quality report to {cfg.data.quality_report}")

    logger.info("Creating engine splits (80 dev / 20 val, 5 folds)...")
    train_engines = sorted(train_df["unit_id"].unique().tolist())
    split_manifest = create_engine_splits(
        train_engines,
        dev_count=cfg.splits.dev_engines_count,
        val_count=cfg.splits.val_engines_count,
        n_folds=cfg.splits.n_folds,
        seed=cfg.splits.seed,
    )
    save_split_manifest(split_manifest, Path(cfg.data.split_manifest))
    logger.info(f"Saved split manifest to {cfg.data.split_manifest}")

    logger.info("Creating frozen snapshot manifest for validation engines...")
    snapshot_manifest = create_snapshot_manifest(
        train_df,
        val_engines=split_manifest.val_engines,
        offsets=cfg.splits.val_snapshot_offsets,
        min_history=cfg.splits.min_history_cycles,
        seed=cfg.splits.seed,
    )
    save_snapshot_manifest(snapshot_manifest, Path(cfg.data.snapshot_manifest))
    logger.info(f"Saved snapshot manifest to {cfg.data.snapshot_manifest}")

    logger.info("Computing RUL labels and proxies for training trajectories...")
    train_labeled = compute_rul_labels(train_df, target_cap=cfg.features.target_cap)
    train_labeled = assign_anomaly_proxies(
        train_labeled,
        high_threshold=cfg.features.high_rul_proxy_threshold,
        near_failure_threshold=cfg.features.near_failure_proxy_threshold,
    )

    dev_df = train_labeled[train_labeled["unit_id"].isin(split_manifest.dev_engines)].copy()
    logger.info("Extracting causal features for 80 development engines...")
    features_dev = extract_causal_features_for_dataframe(
        dev_df,
        window_size=cfg.features.window_size,
        target_cap=cfg.features.target_cap,
        include_labels=True,
        sample_weights_per_engine=True,
    )
    features_dev = assign_anomaly_proxies(
        features_dev,
        high_threshold=cfg.features.high_rul_proxy_threshold,
        near_failure_threshold=cfg.features.near_failure_proxy_threshold,
    )
    features_dev.to_parquet(cfg.data.features_dev, index=False)
    logger.info(f"Saved dev features ({len(features_dev)} rows) to {cfg.data.features_dev}")

    val_df = train_labeled[train_labeled["unit_id"].isin(split_manifest.val_engines)].copy()
    logger.info("Extracting frozen snapshot features for 20 validation engines...")
    features_val = extract_snapshot_features(
        val_df,
        snapshot_cutoffs=snapshot_manifest.val_snapshots,
        window_size=cfg.features.window_size,
        target_cap=cfg.features.target_cap,
    )
    features_val = assign_anomaly_proxies(
        features_val,
        high_threshold=cfg.features.high_rul_proxy_threshold,
        near_failure_threshold=cfg.features.near_failure_proxy_threshold,
    )
    features_val.to_parquet(cfg.data.features_val, index=False)
    logger.info(f"Saved validation snapshot features ({len(features_val)} rows) to {cfg.data.features_val}")

    logger.info("Extracting causal features for test engines (without labels)...")
    features_test = extract_causal_features_for_dataframe(
        test_df,
        window_size=cfg.features.window_size,
        target_cap=cfg.features.target_cap,
        include_labels=False,
        sample_weights_per_engine=False,
    )
    features_test.to_parquet(cfg.data.features_test, index=False)
    logger.info(f"Saved test features ({len(features_test)} rows) to {cfg.data.features_test}")

    logger.info("Data preparation completed successfully.")


def run_policy_report(config_path: str | Path, split: str = "validation") -> None:
    """Generate detailed policy and worklist ranking report."""
    from turbineguard.evaluate import run_validation_evaluation
    rep = run_validation_evaluation(config_path)
    print("\n--- POLICY & WORKLIST CAPACITY REPORT ---")
    diag = rep["policy_diagnostics"]
    print(f"Split: {split}")
    print(f"Total Engines: {diag['total_engines']}")
    print(f"Capacity k (top 20%): {diag['capacity_k']} engines")
    print(f"Actual Near-Failure Engines (RUL <= 30): {diag['actual_positives_count']} ({diag['actual_prevalence']*100:.1f}%)")
    print(f"Selected Positives in Top-k: {diag['selected_positives_count']}")
    print(f"Precision@k: {diag['precision_at_k']*100:.1f}%")
    print(f"Recall@k: {diag['recall_at_k']*100:.1f}%")
    print(f"Lift over prevalence: {diag['lift_over_prevalence']:.2f}x")
    print("----------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="TurbineGuard CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare subcommand
    prep_parser = subparsers.add_parser("prepare", help="Prepare data splits, snapshots, and features")
    prep_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train baseline or candidate models")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    train_parser.add_argument(
        "--experiment",
        type=str,
        default="baselines",
        choices=["baselines", "xgboost", "anomaly"],
        help="Experiment suite to run",
    )

    # freeze subcommand
    freeze_parser = subparsers.add_parser("freeze", help="Freeze validated staging models into versioned bundle")
    freeze_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    freeze_parser.add_argument("--version", type=str, default="v0.1.0", help="Model bundle release version")

    default_bundle = "models/v0.2.0" if Path("models/v0.2.0").exists() else "models/v0.1.0"
    default_release = "v0.2.0" if Path("models/v0.2.0").exists() else "v0.1.0"

    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models on validation or holdout split")
    eval_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    eval_parser.add_argument("--split", type=str, default="validation", choices=["validation", "official-test"], help="Evaluation split")
    eval_parser.add_argument("--bundle", type=str, default=default_bundle, help="Path to frozen model bundle directory")
    eval_parser.add_argument("--release-id", type=str, default=default_release, help="Release identifier")
    eval_parser.add_argument("--allow-heldout-evaluation", action="store_true", help="Explicit consent flag for official test set evaluation")

    # score subcommand
    score_parser = subparsers.add_parser("score", help="Score engine trajectories and generate ranked worklist")
    score_parser.add_argument("--bundle", type=str, default=default_bundle, help="Path to model bundle directory")
    score_parser.add_argument("--input", type=str, required=True, help="Path to raw CMAPSS or CSV trajectory file")
    score_parser.add_argument("--output", type=str, required=True, help="Output CSV path for ranked worklist")
    score_parser.add_argument("--format", "--input-format", dest="format", type=str, default="cmapss", choices=["cmapss", "csv"], help="Input format")

    # explain subcommand
    exp_parser = subparsers.add_parser("explain", help="Generate SHAP feature attributions for engine trajectories")
    exp_parser.add_argument("--bundle", type=str, default=default_bundle, help="Path to model bundle directory")
    exp_parser.add_argument("--input", type=str, required=True, help="Path to raw CMAPSS or CSV trajectory file")
    exp_parser.add_argument("--output", type=str, required=True, help="Output JSON path for explanations")
    exp_parser.add_argument("--format", "--input-format", dest="format", type=str, default="cmapss", choices=["cmapss", "csv"], help="Input format")
    exp_parser.add_argument("--top-k", type=int, default=5, help="Number of top contributing features per engine")

    # drift subcommand
    drift_parser = subparsers.add_parser("drift", help="Run Evidently drift analysis against monitoring reference")
    drift_parser.add_argument("--bundle", type=str, default=default_bundle, help="Path to model bundle directory")
    drift_parser.add_argument("--current", type=str, required=True, help="Path to current cohort features parquet file")
    drift_parser.add_argument("--output", type=str, default="reports/drift", help="Directory to save drift report")

    # drift-controls subcommand
    dc_parser = subparsers.add_parser("drift-controls", help="Execute controlled drift validation tests")
    dc_parser.add_argument("--bundle", type=str, default=default_bundle, help="Path to model bundle directory")
    dc_parser.add_argument("--output", type=str, default="reports/drift/controls", help="Directory to save drift control results")

    # policy-report subcommand
    policy_parser = subparsers.add_parser("policy-report", help="Generate policy and capacity report")
    policy_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    policy_parser.add_argument("--split", type=str, default="validation", help="Split to analyze")

    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare(args.config)
    elif args.command == "train":
        if args.experiment == "baselines":
            from turbineguard.train import run_baseline_experiments
            run_baseline_experiments(args.config)
        elif args.experiment == "xgboost":
            from turbineguard.train import run_xgboost_experiments
            run_xgboost_experiments(args.config)
        elif args.experiment == "anomaly":
            from turbineguard.anomaly import fit_isolation_forest_component
            dev_df = pd.read_parquet(load_config(args.config).data.features_dev)
            fit_isolation_forest_component(dev_df, load_config(args.config))
        else:
            raise NotImplementedError(f"Experiment '{args.experiment}' not recognized.")
    elif args.command == "freeze":
        from turbineguard.artifacts import freeze_model_bundle
        freeze_model_bundle(config_path=args.config, version=args.version)
    elif args.command == "evaluate":
        if args.split == "validation":
            from turbineguard.evaluate import run_validation_evaluation
            run_validation_evaluation(args.config)
        elif args.split == "official-test":
            from turbineguard.evaluate import run_official_test_evaluation
            run_official_test_evaluation(
                config_path=args.config,
                bundle_dir=args.bundle,
                release_id=args.release_id,
                allow_heldout_evaluation=args.allow_heldout_evaluation,
            )
        else:
            raise NotImplementedError(f"Split '{args.split}' not supported.")
    elif args.command == "score":
        from turbineguard.artifacts import load_model_bundle
        from turbineguard.predict import load_and_validate_input_file, score_batch_history
        bundle = load_model_bundle(args.bundle)
        input_df = load_and_validate_input_file(Path(args.input), input_format=args.format)
        ranked_worklist, latest_features = score_batch_history(bundle, input_df)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranked_worklist.to_csv(out_path, index=False)
        feat_path = out_path.parent / "worklist_features.parquet"
        latest_features.to_parquet(feat_path, index=False)
        logger.info(f"Saved scored worklist ({len(ranked_worklist)} engines) to {out_path} and features to {feat_path}")
    elif args.command == "explain":
        import json

        from turbineguard.artifacts import load_model_bundle
        from turbineguard.explain import explain_batch_features
        from turbineguard.predict import load_and_validate_input_file, score_batch_history
        bundle = load_model_bundle(args.bundle)
        input_df = load_and_validate_input_file(Path(args.input), input_format=args.format)
        _, latest_features = score_batch_history(bundle, input_df)
        explanations = explain_batch_features(bundle, latest_features, top_k=args.top_k)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(explanations, indent=2), encoding="utf-8")
        logger.info(f"Saved explanations ({len(explanations)} engines) to {out_path}")
    elif args.command == "drift":
        from turbineguard.artifacts import load_model_bundle
        from turbineguard.monitoring import run_drift_analysis
        bundle = load_model_bundle(args.bundle)
        curr_df = pd.read_parquet(args.current)
        summary = run_drift_analysis(bundle, curr_df, output_dir=args.output)
        logger.info(f"Drift analysis completed. Status: {summary.status}, dataset_drift: {summary.dataset_drift}")
    elif args.command == "drift-controls":
        from turbineguard.artifacts import load_model_bundle
        from turbineguard.monitoring import run_drift_controls
        bundle = load_model_bundle(args.bundle)
        res = run_drift_controls(bundle, output_dir=args.output)
        logger.info(f"Drift controls completed. All passed: {res['all_controls_passed']}")
    elif args.command == "policy-report":
        run_policy_report(args.config, args.split)


if __name__ == "__main__":
    main()

