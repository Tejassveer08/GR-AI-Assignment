# Prediction Page — Churn Prediction Sandbox

The **Prediction** page, also called the **Churn Prediction Sandbox**, lets you test and experiment with the churn prediction model for your e-commerce customers by adjusting customer characteristics using easy-to-understand sliders and dropdowns.

---

## What Can You Do Here?

- **Simulate any customer profile**—as specific or hypothetical as you wish.
- **Instantly see the predicted probability** that this customer will churn (leave your business).
- **Adjust the sensitivity** (decision threshold) to fine-tune whether you want to aggressively or conservatively flag customers as "at risk."
- **Get actionable recommendations** on which customers to prioritize for retention efforts, based on probability and business costs.

---

## How It Works

### 1. **Customer Profile Sliders and Dropdowns**

Use the provided sliders and select menus to describe any customer or test case. Examples include:

- **Age**: 18 to 80
- **Membership Years**: 0 to 15
- **Monthly Login Frequency**: 0 to 40
- **Avg Session Duration (min)**: 0.00 to 60.00
- **Pages per Session**: 1.00 to 25.00
- **Cart Abandonment Rate (%)**: 0.00 to 100.00
- **Wishlist Items**: 0 to 20
- **Purchases (12m)**: 0.00 to 50.00
- **Average Order Value**: $10.00 to $300.00
- **Days Since Last Purchase**: 0 to 365
- **Discount Usage Rate (%)**: 0.00 to 100.00
- **Returns Rate (%)**: 0.00 to 100.00
- **Email Open Rate (%)**: 0.00 to 100.00
- **Customer Service Calls**: 0 to 20
- **Product Reviews Written**: 0 to 20
- **Social Media Score**: 0.00 to 100.00
- **Mobile App Usage (hrs/month)**: 0.00 to 80.00
- **Payment Method Diversity**: 1 to 6
- **Lifetime Value**: $100.00 to $6,000.00
- **Credit Balance (loyalty points)**: 0.00 to 6,000.00
- **Country**, **City**, **Signup Quarter**, **Membership Stage**, **Seasonal Activity**: Choose from available options

You can mix and match any combination to reflect real customers or business scenarios.

---

### 2. **Prediction and Decision Details**

After selecting your inputs, the app will display:

- **Predicted Churn Probability**  
  Example: **64.1%** chance this customer will churn.

- **Decision Threshold (Slider)**  
  Adjust where you “draw the line” for classifying a customer as “churn.”
  - **Lower threshold:** More customers flagged as "churn" (catch more at-risk, but may spend more on campaigns).
  - **Higher threshold:** Fewer false positives, but you may miss some at-risk customers.

- **Recommendation**  
  Clearly states whether the model advises to **Churn** (take retention action) or **Retain** (no action needed), based on your chosen threshold.

- **Business Context and Costs**  
  Example: "$500 loss for missed churners, $50 cost for unnecessary retention campaigns." Cost settings can be tuned for your business model.

---

## How to Analyze Results

- **What-If Testing:** Change sliders to see how different factors impact churn. Useful for understanding which behaviors are risky.
- **Strategy Planning:** Use your most common customer types—see who is most at risk and how to better target retention offers.
- **Threshold Tuning:** Pair with cost-sensitive analysis to set campaign cutoffs that balance risk and budget.

---

## Example Use Case

> A 35-year-old, "Newbie" member in Bangalore who rarely logs in, has high cart abandonment, and a low purchase number is predicted to have a **64.1% chance of churning**. With a threshold of 0.5, the recommendation will be "Churn"—meaning you should probably act to keep this customer.

---

## Summary

- **Test any customer scenario** — see their churn probability instantly.
- **Customize features with sliders and dropdowns.**
- **Adjust decision threshold for your business needs.**
- **Get clear, actionable model recommendations.**
- **Understand how your data drives churn — and build smarter retention tactics.**

This page turns your customer analytics into an interactive, business-ready decision tool.
