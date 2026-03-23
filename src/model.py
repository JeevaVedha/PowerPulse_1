import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate_models(X, y, tune_hyperparams=False):
    """
    Trains and evaluates multiple regression models:
    - Linear Regression (with scaling)
    - Random Forest
    - Gradient Boosting

    Args:
        X: Feature matrix
        y: Target variable
        tune_hyperparams: If True, runs RandomizedSearchCV on Random Forest

    Returns:
        results (dict): MAE, RMSE, R2, y_test, y_pred for each model
    """

    # Train/test split — shuffle=False preserves time order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    results = {}

    # ─────────────────────────────────────────────
    # 1. Linear Regression (with StandardScaler)
    # ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    results["Linear Regression"] = {
        "MAE":    mean_absolute_error(y_test, lr_pred),
        "RMSE":   np.sqrt(mean_squared_error(y_test, lr_pred)),
        "R2":     r2_score(y_test, lr_pred),
        "y_test": y_test,
        "y_pred": lr_pred
    }

    # ─────────────────────────────────────────────
    # 2. Random Forest (with optional tuning)
    # ─────────────────────────────────────────────
    if tune_hyperparams:
        print("Running Hyperparameter Tuning for Random Forest...")
        param_dist = {
            'n_estimators':      [50, 100, 200],
            'max_depth':         [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_distributions=param_dist,
            n_iter=10,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1
        )
        rf_search.fit(X_train, y_train)
        rf = rf_search.best_estimator_
        print("Best RF params:", rf_search.best_params_)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    results["Random Forest"] = {
        "MAE":    mean_absolute_error(y_test, rf_pred),
        "RMSE":   np.sqrt(mean_squared_error(y_test, rf_pred)),
        "R2":     r2_score(y_test, rf_pred),
        "y_test": y_test,
        "y_pred": rf_pred,
        "model":  rf,
        "feature_names": list(X.columns)
    }

    # ─────────────────────────────────────────────
    # 3. Gradient Boosting
    # ─────────────────────────────────────────────
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    results["Gradient Boosting"] = {
        "MAE":    mean_absolute_error(y_test, gb_pred),
        "RMSE":   np.sqrt(mean_squared_error(y_test, gb_pred)),
        "R2":     r2_score(y_test, gb_pred),
        "y_test": y_test,
        "y_pred": gb_pred,
        "model":  gb,
        "feature_names": list(X.columns)
    }

    return results


def plot_feature_importance(results, top_n=10):
    """
    Plots feature importance for tree-based models (RF and GB).
    Saves as feature_importance.png
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_name in zip(axes, ["Random Forest", "Gradient Boosting"]):
        if model_name not in results:
            continue
        model         = results[model_name]["model"]
        feature_names = results[model_name]["feature_names"]
        importances   = model.feature_importances_

        # Sort top N
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values   = importances[indices]

        ax.barh(top_features[::-1], top_values[::-1], color='steelblue')
        ax.set_title(f'{model_name} — Feature Importance')
        ax.set_xlabel('Importance Score')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    plt.close()
    print("Saved: feature_importance.png")


def plot_model_comparison(results):
    """
    Bar chart comparing MAE, RMSE, R2 across all models.
    Saves as model_comparison.png
    """

    models = list(results.keys())
    mae_vals  = [results[m]['MAE']  for m in models]
    rmse_vals = [results[m]['RMSE'] for m in models]
    r2_vals   = [results[m]['R2']   for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(x, mae_vals, width=0.5, color=['steelblue', 'coral', 'teal'])
    axes[0].set_title('MAE Comparison')
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15)

    axes[1].bar(x, rmse_vals, width=0.5, color=['steelblue', 'coral', 'teal'])
    axes[1].set_title('RMSE Comparison')
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15)

    axes[2].bar(x, r2_vals, width=0.5, color=['steelblue', 'coral', 'teal'])
    axes[2].set_title('R² Score Comparison')
    axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=15)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100)
    plt.close()
    print("Saved: model_comparison.png")