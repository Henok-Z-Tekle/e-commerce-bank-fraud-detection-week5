import argparse
from pathlib import Path

from src.cleaning import clean_creditcard_data, clean_fraud_data
from src.eda import plot_bivariate, plot_class_distribution, plot_univariate
from src.features import add_time_features, add_user_velocity_features
from src.geolocation import analyze_fraud_by_country, merge_ip_country
from src.transform import (
    apply_preprocessor,
    build_preprocessor,
    class_distribution,
    resample_training_data,
    split_train_test,
)
from src.utils import load_csv


def run(args):
    fraud_df = load_csv(args.fraud_path)
    credit_df = load_csv(args.creditcard_path)
    ip_df = load_csv(args.ip_path)

    fraud_df = clean_fraud_data(fraud_df)
    credit_df = clean_creditcard_data(credit_df)

    fig_dir = Path(args.output_dir) / "figures"

    plot_univariate(fraud_df, "purchase_value", "Fraud: Purchase Value", fig_dir / "fraud_purchase_value.png")
    plot_univariate(fraud_df, "age", "Fraud: Age", fig_dir / "fraud_age.png")
    plot_bivariate(fraud_df, "purchase_value", "class", "Fraud: Value by Class", fig_dir / "fraud_value_by_class.png")
    plot_class_distribution(fraud_df, "class", "Fraud: Class Distribution", fig_dir / "fraud_class_dist.png")

    plot_univariate(credit_df, "Amount", "Credit: Amount", fig_dir / "credit_amount.png")
    plot_bivariate(credit_df, "Amount", "Class", "Credit: Amount by Class", fig_dir / "credit_amount_by_class.png")
    plot_class_distribution(credit_df, "Class", "Credit: Class Distribution", fig_dir / "credit_class_dist.png")

    fraud_geo = merge_ip_country(fraud_df, ip_df)
    fraud_by_country = analyze_fraud_by_country(fraud_geo, target_col="class")
    fraud_by_country.to_csv(Path(args.output_dir) / "fraud_by_country.csv", index=False)

    fraud_geo = add_time_features(fraud_geo)
    fraud_geo = add_user_velocity_features(fraud_geo)

    X_train, X_test, y_train, y_test = split_train_test(fraud_geo, target_col="class")
    preprocessor, _, _ = build_preprocessor(X_train)
    X_train_t, X_test_t = apply_preprocessor(preprocessor, X_train, X_test)

    dist_before = class_distribution(y_train)
    dist_before.to_csv(Path(args.output_dir) / "fraud_class_dist_before.csv")

    X_res, y_res = resample_training_data(X_train_t, y_train, method=args.resample)
    dist_after = class_distribution(y_res)
    dist_after.to_csv(Path(args.output_dir) / "fraud_class_dist_after.csv")

    Xc_train, Xc_test, yc_train, yc_test = split_train_test(credit_df, target_col="Class")
    cc_preprocessor, _, _ = build_preprocessor(Xc_train)
    Xc_train_t, Xc_test_t = apply_preprocessor(cc_preprocessor, Xc_train, Xc_test)

    dist_before_cc = class_distribution(yc_train)
    dist_before_cc.to_csv(Path(args.output_dir) / "credit_class_dist_before.csv")

    Xc_res, yc_res = resample_training_data(Xc_train_t, yc_train, method=args.resample)
    dist_after_cc = class_distribution(yc_res)
    dist_after_cc.to_csv(Path(args.output_dir) / "credit_class_dist_after.csv")

    print("Preprocessing complete. Outputs written to", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud data preprocessing pipeline")
    parser.add_argument("--fraud-path", required=True, help="Path to Fraud_Data.csv")
    parser.add_argument("--creditcard-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--ip-path", required=True, help="Path to IpAddress_to_Country.csv")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--resample", default="undersample", choices=["smote", "undersample"], help="Resampling method")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
