# main.py
# DO NOT MODIFY THIS FILE

import json
from solve import (
    load_data,
    ols_with_intercept,
    ols_no_intercept,
    predict_with_intercept,
    predict_no_intercept,
    compute_metrics
)

def main():

    # Load dataset
    X_train, y_train, X_test, y_test = load_data()

    # -----------------------------
    # Standard OLS (with intercept)
    # -----------------------------
    w, w0 = ols_with_intercept(X_train, y_train)

    yhat_train = predict_with_intercept(X_train, w, w0)
    yhat_test = predict_with_intercept(X_test, w, w0)

    train_metrics = compute_metrics(y_train, yhat_train)
    test_metrics = compute_metrics(y_test, yhat_test)

    # -----------------------------
    # OLS without intercept
    # -----------------------------
    w_no = ols_no_intercept(X_train, y_train)
    
    yhat_train_no = predict_no_intercept(X_train, w_no)
    yhat_test_no = predict_no_intercept(X_test, w_no)

    train_metrics_no = compute_metrics(y_train, yhat_train_no)
    test_metrics_no = compute_metrics(y_test, yhat_test_no)

    results = {
        "OLS_with_intercept": {
            "train": train_metrics,
            "test": test_metrics
        },
        "OLS_no_intercept": {
            "train": train_metrics_no,
            "test": test_metrics_no
        }
    }

    print(json.dumps(results, indent=4))

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()