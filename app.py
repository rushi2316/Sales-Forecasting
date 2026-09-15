import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet


st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting Dashboard")
st.markdown("### Store Sales Prediction using Machine Learning")


@st.cache_data
def load_data():

    zip_file = "store-sales-time-series-forecasting (1).zip"

    with zipfile.ZipFile(zip_file, "r") as z:

        train_file = [
            f for f in z.namelist()
            if f.endswith("train.csv")
        ][0]

        with z.open(train_file) as f:
            df = pd.read_csv(f)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df.rename(
        columns={
            "date": "Date",
            "sales": "Sales"
        },
        inplace=True
    )

    return df


@st.cache_data
def prepare_data(store, family):

    df = load_data()

    df = df[
        (df["store_nbr"] == store) &
        (df["family"] == family)
    ].copy()

    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["lag_1"] = df["Sales"].shift(1)
    df["lag_7"] = df["Sales"].shift(7)
    df["rolling_mean_7"] = df["Sales"].rolling(7).mean()

    return df.dropna()


features = [
    "dayofweek",
    "month",
    "year",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]


@st.cache_resource
def train_models(store, family):

    df = prepare_data(store, family)

    if len(df) < 20:
        return None

    lr = LinearRegression()

    rf = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    xg = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    lr.fit(df[features], df["Sales"])
    rf.fit(df[features], df["Sales"])
    xg.fit(df[features], df["Sales"])

    return lr, rf, xg


df = load_data()


st.sidebar.header("Prediction Settings")

stores = sorted(df["store_nbr"].unique())
families = sorted(df["family"].unique())

store = st.sidebar.selectbox(
    "Select Store",
    stores
)

family = st.sidebar.selectbox(
    "Select Product Family",
    families
)


st.sidebar.header("Project Settings")

st.sidebar.write(f"Store: {store}")
st.sidebar.write(f"Product Family: {family}")


selected_df = df[
    (df["store_nbr"] == store) &
    (df["family"] == family)
].copy()


model_df = prepare_data(store, family)


if st.sidebar.button("Run Forecasting Model"):

    if len(model_df) < 20:

        st.error(
            "Not enough data for this selection."
        )

    else:

        train = model_df[
            model_df["Date"] < "2017-01-01"
        ]

        test = model_df[
            model_df["Date"] >= "2017-01-01"
        ]

        if len(train) == 0 or len(test) == 0:

            st.error(
                "Not enough training or testing data."
            )

        else:

            X_train = train[features]
            y_train = train["Sales"]

            X_test = test[features]
            y_test = test["Sales"]


            models = train_models(store, family)

            lr, rf, xg = models


            lr_pred = lr.predict(X_test)
            rf_pred = rf.predict(X_test)
            xgb_pred = xg.predict(X_test)


            final_pred = (
                lr_pred +
                rf_pred +
                xgb_pred
            ) / 3


            st.success(
                "Models trained successfully!"
            )


            st.header("📊 Model Performance")


            results = pd.DataFrame({

                "Model": [
                    "Linear Regression",
                    "Random Forest",
                    "XGBoost",
                    "Ensemble"
                ],

                "MAE": [
                    mean_absolute_error(
                        y_test,
                        lr_pred
                    ),

                    mean_absolute_error(
                        y_test,
                        rf_pred
                    ),

                    mean_absolute_error(
                        y_test,
                        xgb_pred
                    ),

                    mean_absolute_error(
                        y_test,
                        final_pred
                    )
                ],

                "RMSE": [
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            lr_pred
                        )
                    ),

                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            rf_pred
                        )
                    ),

                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            xgb_pred
                        )
                    ),

                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            final_pred
                        )
                    )
                ]
            })


            st.dataframe(
                results.style.format({
                    "MAE": "{:.2f}",
                    "RMSE": "{:.2f}"
                }),
                use_container_width=True
            )


            st.header("📈 Actual vs Predicted Sales")


            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.plot(
                test["Date"],
                y_test,
                label="Actual"
            )

            ax.plot(
                test["Date"],
                xgb_pred,
                label="XGBoost"
            )

            ax.set_title(
                "Actual vs Predicted Sales"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()

            st.pyplot(fig)


            st.header("🤖 Feature Importance")


            importance = pd.Series(
                xg.feature_importances_,
                index=features
            ).sort_values()


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            importance.plot(
                kind="barh",
                ax=ax
            )

            ax.set_title(
                "XGBoost Feature Importance"
            )

            st.pyplot(fig)


            st.header("📊 Model Comparison")


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.bar(
                results["Model"],
                results["MAE"]
            )

            ax.set_title(
                "Model Comparison - MAE"
            )

            ax.set_ylabel("MAE")

            plt.xticks(rotation=20)

            st.pyplot(fig)


            st.header("🔥 Feature Correlation")


            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            sns.heatmap(
                model_df[
                    features + ["Sales"]
                ].corr(),
                annot=True,
                ax=ax
            )

            ax.set_title(
                "Feature Correlation"
            )

            st.pyplot(fig)


            st.header("🔮 30-Day Forecast")


            prophet_df = model_df[
                ["Date", "Sales"]
            ].rename(
                columns={
                    "Date": "ds",
                    "Sales": "y"
                }
            )


            prophet_model = Prophet()

            prophet_model.fit(
                prophet_df
            )


            future = prophet_model.make_future_dataframe(
                periods=30
            )


            forecast = prophet_model.predict(
                future
            )


            future_forecast = forecast[
                [
                    "ds",
                    "yhat",
                    "yhat_lower",
                    "yhat_upper"
                ]
            ].tail(30)


            st.dataframe(
                future_forecast,
                use_container_width=True
            )


            fig = prophet_model.plot(
                forecast
            )

            st.pyplot(fig)


            st.header("📋 Recent Sales Data")


            st.dataframe(
                model_df.tail(20),
                use_container_width=True
            )


else:

    st.info(
        "Select a Store and Product Family, "
        "then click 'Run Forecasting Model'."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        selected_df.head(10),
        use_container_width=True
    )


st.divider()

st.header("🔮 Predict Your Own Sales")

st.write(
    f"Prediction for Store {store} - {family}"
)


prediction_date = st.date_input(
    "Select Prediction Date"
)


previous_sales = st.number_input(
    "Previous Day Sales",
    min_value=0.0,
    value=2500.0
)


sales_7_days = st.number_input(
    "Sales 7 Days Ago",
    min_value=0.0,
    value=2200.0
)


average_sales = st.number_input(
    "7-Day Average Sales",
    min_value=0.0,
    value=2350.0
)


if st.button("🔮 Predict Sales"):

    if len(model_df) < 20:

        st.error(
            "Not enough historical data."
        )

    else:

        models = train_models(
            store,
            family
        )

        lr, rf, xg = models


        date_value = pd.to_datetime(
            prediction_date
        )


        X = pd.DataFrame({

            "dayofweek": [
                date_value.dayofweek
            ],

            "month": [
                date_value.month
            ],

            "year": [
                date_value.year
            ],

            "lag_1": [
                previous_sales
            ],

            "lag_7": [
                sales_7_days
            ],

            "rolling_mean_7": [
                average_sales
            ]
        })


        lr_prediction = lr.predict(X)[0]

        rf_prediction = rf.predict(X)[0]

        xgb_prediction = xg.predict(X)[0]


        ensemble_prediction = (
            lr_prediction +
            rf_prediction +
            xgb_prediction
        ) / 3


        st.success(
            "Prediction completed successfully!"
        )


        st.subheader(
            "📊 Prediction Results"
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Linear Regression",
            f"{lr_prediction:,.2f}"
        )


        col2.metric(
            "Random Forest",
            f"{rf_prediction:,.2f}"
        )


        col3.metric(
            "XGBoost",
            f"{xgb_prediction:,.2f}"
        )


        col4.metric(
            "Ensemble Prediction",
            f"{ensemble_prediction:,.2f}"
        )


        st.divider()


        st.subheader(
            "🎯 Final Predicted Sales"
        )


        st.metric(
            "Predicted Sales",
            f"{ensemble_prediction:,.2f}"
        )


        result = pd.DataFrame({

            "Prediction Date": [
                prediction_date
            ],

            "Store": [
                store
            ],

            "Product Family": [
                family
            ],

            "Previous Day Sales": [
                previous_sales
            ],

            "Sales 7 Days Ago": [
                sales_7_days
            ],

            "7-Day Average Sales": [
                average_sales
            ],

            "Linear Regression": [
                lr_prediction
            ],

            "Random Forest": [
                rf_prediction
            ],

            "XGBoost": [
                xgb_prediction
            ],

            "Ensemble Prediction": [
                ensemble_prediction
            ]
        })


        st.download_button(
            "📥 Download Prediction",
            result.to_csv(index=False),
            "sales_prediction.csv",
            "text/csv"
        )