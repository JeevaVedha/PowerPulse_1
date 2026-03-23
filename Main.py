from src import (
    load_and_preprocess_power_data,
    create_time_features,
    create_lag_features,
    train_and_evaluate_models,
    plot_feature_importance,
    plot_model_comparison
)


def main():

    # ── Step 1: Load & Preprocess ──────────────────────────────────
    print("=" * 50)
    print("Step 1: Loading & Preprocessing Data...")
    print("=" * 50)
    df = load_and_preprocess_power_data()

    # ── Step 2: Feature Engineering ───────────────────────────────
    print("=" * 50)
    print("Step 2: Creating Time Features...")
    print("=" * 50)
    df = create_time_features(df)

    print("Step 3: Creating Lag Features...")
    df = create_lag_features(df)

    # ── Step 3: Prepare X, y ──────────────────────────────────────
    if "DateTime" in df.columns:
        df = df.drop("DateTime", axis=1)

    target_column = "Global_active_power"
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"\nFeatures used ({len(X.columns)}): {list(X.columns)}")

    # ── Step 4: Train Models ──────────────────────────────────────
    print("=" * 50)
    print("Step 4: Training Models...")
    print("=" * 50)
    results = train_and_evaluate_models(X, y, tune_hyperparams=False)

    # ── Step 5: Print Results ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("Model Results:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  RMSE : {metrics['RMSE']:.4f}")
        print(f"  R2   : {metrics['R2']:.4f}")

    # ── Step 6: Visualizations ────────────────────────────────────
    print("\nGenerating plots...")
    plot_feature_importance(results)
    plot_model_comparison(results)
    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()