"""
Anaerobic Digestion ML Analysis
Implements the workflow described in the supplied Chapters 3 and 4:
- Load and inspect experimental CSV data
- Clean missing/duplicate records
- Encode inoculum diversity and scale numeric inputs
- Exploratory analysis and correlation plots
- 80/20 train-test split
- Linear Regression, Random Forest, SVR and XGBoost models
- GridSearchCV tuning
- R2, MAE and RMSE evaluation
- Model comparison, feature importance and partial dependence plots
- Save trained models and prediction outputs

IMPORTANT:
The supplied methodology mentions SMOTE. Standard SMOTE is designed for
classification with categorical class labels. It should not be applied directly
to continuous regression targets such as methane yield or nutrient concentrations.
This script therefore does not apply standard SMOTE. If regression augmentation
is required, assess a regression-specific method (for example, SMOGN) separately
and only within the training folds to prevent data leakage.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

from xgboost import XGBRegressor


# ---------------------------------------------------------------------
# 1. CONFIGURATION — EDIT THESE VALUES TO MATCH YOUR CSV FILE
# ---------------------------------------------------------------------

DATA_FILE = "anaerobic_digestion_data.csv"
OUTPUT_DIR = Path("anaerobic_digestion_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Change these names if the headings in your CSV are different.
FEATURES = ["TS", "ID", "AC"]
TARGETS = ["CMY", "N", "P", "K", "Zn"]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Set to True only if you want to remove rows with extreme numeric input
# values using the IQR rule. Review the removed rows before using this.
REMOVE_INPUT_OUTLIERS = False


# ---------------------------------------------------------------------
# 2. LOAD AND INSPECT DATA
# ---------------------------------------------------------------------

def load_data():
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. Put your experimental CSV in the "
            "same folder as this script or update DATA_FILE."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    required = FEATURES + TARGETS
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The following required columns were not found in the CSV: "
            f"{missing_columns}\nAvailable columns: {list(df.columns)}"
        )

    print("\nDATA OVERVIEW")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df[required].isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().copy()

    # Ensure numeric process variables and output measurements are numeric.
    for col in ["TS", "AC"] + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ID may be numeric-coded or categorical. Treat it as categorical.
    df["ID"] = df["ID"].astype("string").str.strip()
    df["ID"] = df["ID"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Rows without any target measurement cannot be used for that target.
    # Missing feature values are imputed inside each model pipeline.
    return df


# ---------------------------------------------------------------------
# 3. OPTIONAL INPUT OUTLIER FILTER
# ---------------------------------------------------------------------

def remove_iqr_outliers(df):
    numeric_columns = ["TS", "AC"]
    keep = pd.Series(True, index=df.index)

    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        keep &= df[col].isna() | df[col].between(lower, upper)

    removed = int((~keep).sum())
    print(f"\nRows removed by optional IQR input filter: {removed}")
    return df.loc[keep].copy()


# ---------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------

def perform_eda(df):
    analysis_columns = ["TS", "AC"] + TARGETS
    numeric_df = df[analysis_columns].copy()

    print("\nDESCRIPTIVE STATISTICS")
    print(numeric_df.describe().T)
    numeric_df.describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    # Histograms
    numeric_df.hist(figsize=(14, 10), bins=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "variable_distributions.png", dpi=300)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 5. PREPROCESSING AND MODEL DEFINITIONS
# ---------------------------------------------------------------------

def make_preprocessor():
    numeric_features = ["TS", "AC"]
    categorical_features = ["ID"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


def model_definitions():
    """Return estimators and parameter grids for GridSearchCV."""
    return {
        "Linear Regression": (
            LinearRegression(),
            {}
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 4]
            }
        ),
        "SVR": (
            SVR(kernel="rbf"),
            {
                "model__C": [1, 10, 100],
                "model__epsilon": [0.01, 0.1, 0.2],
                "model__gamma": ["scale", "auto"]
            }
        ),
        "XGBoost": (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        )
    }


# ---------------------------------------------------------------------
# 6. TRAIN, TUNE AND EVALUATE ONE MODEL FOR EACH OUTPUT
# ---------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    # R2 can be undefined for a test set containing fewer than 2 observations.
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def train_and_evaluate(df):
    results = []
    best_models = {}
    prediction_data = pd.DataFrame(index=df.index)

    for target in TARGETS:
        target_df = df[FEATURES + [target]].dropna(subset=[target]).copy()

        if len(target_df) < 6:
            warnings.warn(
                f"Skipping {target}: only {len(target_df)} usable rows. "
                "More observations are needed for a meaningful train/test split."
            )
            continue

        X = target_df[FEATURES]
        y = target_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # Reduce CV folds if the training set is small.
        folds = min(CV_FOLDS, len(X_train))
        if folds < 2:
            warnings.warn(f"Skipping {target}: insufficient training rows for CV.")
            continue

        cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

        for model_name, (estimator, parameter_grid) in model_definitions().items():
            pipeline = Pipeline(steps=[
                ("preprocessor", make_preprocessor()),
                ("model", estimator)
            ])

            if parameter_grid:
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=parameter_grid,
                    scoring="r2",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise"
                )
                search.fit(X_train, y_train)
                fitted_model = search.best_estimator_
                best_params = search.best_params_
            else:
                fitted_model = pipeline.fit(X_train, y_train)
                best_params = {}

            predictions = fitted_model.predict(X_test)
            r2, mae, rmse = regression_metrics(y_test, predictions)

            results.append({
                "Target": target,
                "Model": model_name,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "Best Parameters": str(best_params)
            })

            print(
                f"{target} | {model_name}: "
                f"R2={r2:.3f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
            )

            # Save predictions for each model.
            prediction_data.loc[X_test.index, f"{target}_{model_name}_Predicted"] = predictions

            # Retain the best model per target based on test-set R2.
            current_key = (target, model_name)
            best_models[current_key] = fitted_model

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No models were trained. Check the dataset and column names.")

    results_df.to_csv(OUTPUT_DIR / "model_evaluation_results.csv", index=False)
    prediction_data.to_csv(OUTPUT_DIR / "test_predictions.csv", index_label="row_index")

    # Select the highest test R2 model for each target.
    best_rows = (
        results_df.sort_values(["Target", "R2"], ascending=[True, False])
        .groupby("Target", as_index=False)
        .first()
    )
    best_rows.to_csv(OUTPUT_DIR / "best_model_per_target.csv", index=False)

    for _, row in best_rows.iterrows():
        key = (row["Target"], row["Model"])
        model = best_models[key]
        safe_target = str(row["Target"]).replace("/", "_")
        joblib.dump(model, OUTPUT_DIR / f"best_model_{safe_target}.joblib")

    return results_df, best_rows, best_models


# ---------------------------------------------------------------------
# 7. FEATURE IMPORTANCE AND PARTIAL DEPENDENCE FOR XGBOOST
# ---------------------------------------------------------------------

def explain_xgboost_models(df, best_rows, best_models):
    for _, row in best_rows.iterrows():
        target = row["Target"]
        model_name = row["Model"]
        if model_name != "XGBoost":
            continue

        model = best_models[(target, model_name)]
        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        importances = xgb_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        importance_df.to_csv(
            OUTPUT_DIR / f"{target}_xgboost_feature_importance.csv",
            index=False
        )

        plt.figure(figsize=(9, 5))
        sns.barplot(data=importance_df, x="Importance", y="Feature")
        plt.title(f"XGBoost Feature Importance — {target}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{target}_feature_importance.png", dpi=300)
        plt.close()

        # Partial dependence is plotted against the original input columns.
        X_for_pdp = df[FEATURES].dropna(subset=["TS", "AC"])
        if len(X_for_pdp) > 1:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_for_pdp,
                    features=["TS", "AC"],
                    ax=ax
                )
                plt.suptitle(f"Partial Dependence — {target}")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"{target}_partial_dependence.png", dpi=300)
                plt.close()
            except Exception as exc:
                print(f"Could not create PDP for {target}: {exc}")


# ---------------------------------------------------------------------
# 8. MODEL COMPARISON CHARTS
# ---------------------------------------------------------------------

def plot_model_comparison(results_df):
    for metric in ["R2", "MAE", "RMSE"]:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df,
            x="Target",
            y=metric,
            hue="Model"
        )
        plt.title(f"Model Comparison — {metric}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"model_comparison_{metric.lower()}.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# 9. PREDICTION FOR A NEW OPERATING CONDITION
# ---------------------------------------------------------------------

def predict_new_conditions(model_file, ts, inoculum_diversity, activated_carbon):
    """
    Example:
    predict_new_conditions(
        OUTPUT_DIR / "best_model_CMY.joblib",
        ts=24,
        inoculum_diversity="High",
        activated_carbon=15
    )

    Use the same ID category labels/codes that appear in the training CSV.
    """
    model = joblib.load(model_file)
    new_data = pd.DataFrame([{
        "TS": ts,
        "ID": str(inoculum_diversity),
        "AC": activated_carbon
    }])
    return float(model.predict(new_data)[0])


# ---------------------------------------------------------------------
# 10. MAIN EXECUTION
# ---------------------------------------------------------------------

def main():
    df = load_data()

    if REMOVE_INPUT_OUTLIERS:
        df = remove_iqr_outliers(df)

    perform_eda(df)

    results_df, best_rows, best_models = train_and_evaluate(df)
    explain_xgboost_models(df, best_rows, best_models)
    plot_model_comparison(results_df)

    print("\nMODEL EVALUATION SUMMARY")
    print(results_df.sort_values(["Target", "R2"], ascending=[True, False]).to_string(index=False))

    print("\nBEST MODEL PER TARGET")
    print(best_rows[["Target", "Model", "R2", "MAE", "RMSE"]].to_string(index=False))

    print(f"\nCompleted. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
