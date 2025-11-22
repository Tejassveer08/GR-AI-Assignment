import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="E-Commerce Customer Intelligence", layout="wide")

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_lightgbm.pkl"
SCALER_PATH = ARTIFACT_DIR / "standard_scaler.pkl"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.pkl"
MODEL_DATA_PATH = ARTIFACT_DIR / "model_ready_dataset.csv"
RAW_DATA_PATH = "ecommerce_customer_churn_dataset.csv"


@st.cache_data
def load_data():
    raw_df = pd.read_csv(RAW_DATA_PATH)
    model_df = pd.read_csv(MODEL_DATA_PATH)
    return raw_df, model_df


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


def preprocess_inputs(customer_inputs, features, scaler):
    base_vector = pd.Series({f: 0 for f in features})

    # Fill provided values into base_vector
    for key, value in customer_inputs.items():
        if key in base_vector:
            base_vector[key] = value
        else:
            # For encoded categorical values
            matching_cols = [col for col in features if col.startswith(key + "_")]
            if matching_cols:
                for col in matching_cols:
                    target_value = col.split("_")[-1]
                    base_vector[col] = 1 if str(value) == target_value else 0

    
    base_vector = base_vector.reindex(scaler.feature_names_in_, fill_value=0)

    scaled_vector = scaler.transform([base_vector.values])[0]
    return scaled_vector.reshape(1, -1)



def render_eda(raw_df):
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(raw_df):,}")
    c2.metric("Churn Rate", f"{raw_df['Churned'].mean():.1%}")
    c3.metric("Avg Lifetime Value", f"${raw_df['Lifetime_Value'].mean():,.0f}")

    st.write("Preview")
    st.dataframe(raw_df.head(25))

    st.markdown("#### Feature Distributions")
    feature = st.selectbox(
        "Select feature",
        options=['Age', 'Membership_Years', 'Login_Frequency', 'Total_Purchases',
                 'Average_Order_Value', 'Lifetime_Value', 'Credit_Balance']
    )
    fig = px.histogram(raw_df, x=feature, color='Churned', nbins=40,
                       title=f"{feature} Distribution by Churn")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Missing Value Heatmap")
    missing_df = raw_df.isna()
    fig = px.imshow(missing_df.astype(int), color_continuous_scale='Blues',
                    labels=dict(color='Missing (1=yes)'), aspect='auto')
    st.plotly_chart(fig, use_container_width=True)


def render_segments(model_df):
    st.subheader("Customer Segments")
    if "Segment_Name" not in model_df.columns:
        st.info("Segment labels not found in saved dataset.")
        return

    segment_summary = model_df.groupby("Segment_Name").agg({
        "Churned": "mean",
        "Lifetime_Value": "mean",
        "Total_Purchases": "mean",
        "Discount_Usage_Rate": "mean",
        "Returns_Rate": "mean"
    }).rename(columns={"Churned": "Churn_Rate"})

    st.dataframe(segment_summary.style.format({
        "Churn_Rate": "{:.1%}",
        "Lifetime_Value": "${:,.0f}",
        "Total_Purchases": "{:.1f}",
        "Discount_Usage_Rate": "{:.1f}",
        "Returns_Rate": "{:.1f}"
    }))

    fig = px.bar(segment_summary.reset_index(),
                 x="Segment_Name", y="Churn_Rate",
                 title="Churn Rate by Segment",
                 color="Segment_Name",
                 labels={"Churn_Rate": "Churn Rate"})
    st.plotly_chart(fig, use_container_width=True)


def render_prediction(model, scaler, features, model_df, raw_df):
    st.subheader("Churn Prediction Sandbox")

    customer_inputs = {
        "Age": st.slider("Age", 18, 80, int(raw_df["Age"].median())),
        "Membership_Years": st.slider("Membership Years", 0, 15, 3),
        "Login_Frequency": st.slider("Monthly Login Frequency", 0, 40, 12),
        "Session_Duration_Avg": st.slider("Avg Session Duration (min)", 0.0, 60.0, 25.0),
        "Pages_Per_Session": st.slider("Pages per Session", 1.0, 25.0, 8.0),
        "Cart_Abandonment_Rate": st.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 45.0),
        "Wishlist_Items": st.slider("Wishlist Items", 0, 20, 3),
        "Total_Purchases": st.slider("Purchases (12m)", 0.0, 50.0, 12.0),
        "Average_Order_Value": st.slider("Average Order Value", 10.0, 300.0, 120.0),
        "Days_Since_Last_Purchase": st.slider("Days Since Last Purchase", 0, 365, 40),
        "Discount_Usage_Rate": st.slider("Discount Usage Rate (%)", 0.0, 100.0, 35.0),
        "Returns_Rate": st.slider("Returns Rate (%)", 0.0, 100.0, 8.0),
        "Email_Open_Rate": st.slider("Email Open Rate (%)", 0.0, 100.0, 25.0),
        "Customer_Service_Calls": st.slider("Customer Service Calls", 0, 20, 2),
        "Product_Reviews_Written": st.slider("Product Reviews Written", 0, 20, 1),
        "Social_Media_Engagement_Score": st.slider("Social Media Score", 0.0, 100.0, 30.0),
        "Mobile_App_Usage": st.slider("Mobile App Usage (hrs/month)", 0.0, 80.0, 20.0),
        "Payment_Method_Diversity": st.slider("Payment Method Diversity", 1, 6, 2),
        "Lifetime_Value": st.slider("Lifetime Value", 100.0, 6000.0, 1800.0),
        "Credit_Balance": st.slider("Credit Balance (loyalty points)", 0.0, 6000.0, 1000.0),
        "Country": st.selectbox("Country", sorted(raw_df["Country"].dropna().unique())),
        "City": st.selectbox("City", sorted(raw_df["City"].dropna().unique())),
        "Signup_Quarter": st.selectbox("Signup Quarter", raw_df["Signup_Quarter"].dropna().unique()),
        "Membership_Stage": st.selectbox("Membership Stage", ["Newbie", "Growing", "Established", "Veteran"]),
        "Seasonal_Activity": st.selectbox("Seasonal Activity", ["High", "Moderate", "Peak", "Low"])
    }

    # fill engineered ratios based on inputs
    customer_inputs["Recency"] = 365 - customer_inputs["Days_Since_Last_Purchase"]
    customer_inputs["Frequency"] = customer_inputs["Total_Purchases"] / (customer_inputs["Membership_Years"] + 1e-3)
    customer_inputs["Monetary"] = customer_inputs["Average_Order_Value"] * customer_inputs["Total_Purchases"]
    customer_inputs["Engagement_Index"] = np.cbrt(
        customer_inputs["Login_Frequency"] * max(customer_inputs["Session_Duration_Avg"], 1e-3) *
        max(customer_inputs["Pages_Per_Session"], 1e-3)
    )
    customer_inputs["Support_Intensity"] = customer_inputs["Customer_Service_Calls"] / (customer_inputs["Total_Purchases"] + 1)
    customer_inputs["Purchase_to_Login_Ratio"] = customer_inputs["Total_Purchases"] / (customer_inputs["Login_Frequency"] + 1)
    customer_inputs["Discount_to_Order_Value"] = customer_inputs["Discount_Usage_Rate"] / (customer_inputs["Average_Order_Value"] + 1)
    customer_inputs["Return_to_Purchase_Ratio"] = customer_inputs["Returns_Rate"] / (customer_inputs["Total_Purchases"] + 1)
    customer_inputs["Signup_Quarter_Num"] = int(customer_inputs["Signup_Quarter"].replace("Q", "")) if "Signup_Quarter" in customer_inputs else 1
    customer_inputs["Lifetime_Value_Squared"] = customer_inputs["Lifetime_Value"] ** 2
    customer_inputs["High_Returns_Flag"] = int(customer_inputs["Returns_Rate"] > raw_df["Returns_Rate"].median() * 1.5)
    customer_inputs["Cross_Channel_Engagement"] = (
        customer_inputs["Social_Media_Engagement_Score"] * customer_inputs["Mobile_App_Usage"]
    )

    scaled_vector = preprocess_inputs(customer_inputs, features, scaler)
    churn_probability = model.predict_proba(scaled_vector)[0, 1]

    st.metric("Predicted Churn Probability", f"{churn_probability:.1%}")
    threshold = st.slider("Decision Threshold (cost-sensitive)", 0.1, 0.9, 0.5, 0.05)
    prediction = "Churn" if churn_probability >= threshold else "Retain"
    st.success(f"Recommendation: **{prediction}** (threshold={threshold:.2f})")

    st.caption("Cost model: $500 loss for missed churners, $50 cost for unnecessary retention campaigns.")


def render_cost_analysis(model, scaler, features, model_df):
    st.subheader("Cost-Sensitive Optimization")
    # Align model_df features with scaler expected features
    x = model_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    y = model_df["Churned"]
    probs = model.predict_proba(scaler.transform(X))[:, 1]
    thresholds = np.linspace(0.1, 0.9, 25)
    cost_fn = st.number_input("Cost of False Negative ($)", value=500, min_value=100, max_value=2000, step=50)
    cost_fp = st.number_input("Cost of False Positive ($)", value=50, min_value=0, max_value=500, step=10)

    expected_costs = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        cost = fn * cost_fn + fp * cost_fp
        expected_costs.append(cost)

    df_cost = pd.DataFrame({"threshold": thresholds, "expected_cost": expected_costs})
    best_row = df_cost.loc[df_cost["expected_cost"].idxmin()]

    fig = px.line(df_cost, x="threshold", y="expected_cost", title="Expected Loss vs Threshold")
    fig.add_vline(x=best_row["threshold"], line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Optimal Threshold", f"{best_row['threshold']:.2f}")
    st.metric("Minimum Expected Cost", f"${best_row['expected_cost']:,.0f}")


raw_df, model_df = load_data()
model, scaler, features = load_artifacts()

st.title("Customer Segmentation & Churn Intelligence")
st.write(
    "Explore segments, diagnose churn drivers, and run what-if simulations powered "
    "by the trained LightGBM ensemble."
)

tabs = st.tabs(["EDA", "Segments", "Prediction", "Cost & ROI"])

with tabs[0]:
    render_eda(raw_df)

with tabs[1]:
    render_segments(model_df)

with tabs[2]:
    render_prediction(model, scaler, features, model_df, raw_df)

with tabs[3]:
    render_cost_analysis(model, scaler, features, model_df)

