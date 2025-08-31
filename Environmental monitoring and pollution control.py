import pandas as pd
from src.data_preprocessing import load_data, preprocess
from src.model_training import train_and_save_models
from src.evaluation import evaluate_regression, evaluate_classification, save_metrics
from src.visualization import plot_time_series, plot_feature_importance
import os

def main():
    data_path = input("Enter the path to your CSV dataset (e.g., data/air_quality_data.csv): ").strip()
    output_dir = "results"

    print("Loading data:", data_path)
    df = load_data(data_path)

    print("Preprocessing...")
    X, y_reg, y_cls = preprocess(df)

    print("Training models...")
    reg_pipe, cls_pipe, metrics = train_and_save_models(
        X, y_reg, y_cls, output_dir=os.path.join(output_dir, "models")
    )
    print("Evaluating models...")
    y_reg_pred = reg_pipe.predict(X)
    y_cls_pred = cls_pipe.predict(X)

    reg_metrics = evaluate_regression(y_reg, y_reg_pred)
    cls_metrics = evaluate_classification(y_cls, y_cls_pred)

    all_metrics = {
        "regression": reg_metrics,
        "classification": {"accuracy": cls_metrics["accuracy"],
                           "report": cls_metrics["classification_report"]}
    }
    print("Metrics:", all_metrics)
    save_metrics({"regression": reg_metrics, "classification": cls_metrics}, out_dir=output_dir)
    print("Creating plots...")
    datetime_index = df["datetime"] if "datetime" in df.columns else pd.RangeIndex(start=0, stop=len(df))
    if "pm2_5" in df.columns:
        plot_time_series(datetime_index, df["pm2_5"], "PM2.5", out_path=os.path.join(output_dir, "plots", "pm2_5_time_series.png"))

    feature_names = X.columns.tolist()
    plot_feature_importance(reg_pipe, feature_names, out_path=os.path.join(output_dir, "plots", "feature_importance_reg.png"))

    print(f"\n✅ Models saved in {os.path.join(output_dir, 'models')}")
    print(f"✅ Metrics saved at {output_dir}/metrics_summary.csv")
    print("✅ Plots saved inside results/plots/")
    print("Done!")


if __name__ == "__main__":
    main()
