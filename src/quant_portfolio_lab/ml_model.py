import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_random_forest(price_series):

    df = pd.DataFrame(index=price_series.index)

    df["return"] = price_series.pct_change()

    df["mom5"] = price_series.pct_change(5)

    df["mom20"] = price_series.pct_change(20)

    df["vol20"] = (
        df["return"]
        .rolling(20)
        .std()
    )

    # 下一天上涨=1
    df["target"] = (
        df["return"].shift(-1) > 0
    ).astype(int)

    df = df.dropna()

    X = df[
        [
            "mom5",
            "mom20",
            "vol20"
        ]
    ]

    y = df["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False
        )
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    return model, acc