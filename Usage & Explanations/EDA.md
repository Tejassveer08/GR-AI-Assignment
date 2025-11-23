# EDA (Exploratory Data Analysis) Page

The **EDA page** (“Exploratory Data Analysis”) is your entry point for understanding the structure, shape, and quality of your e-commerce customer dataset before diving into more advanced analytics like segmentation or churn prediction.

---

## 1. Dataset Overview

At the top of the page, you’ll find a concise overview of your dataset:

- **Total Customers:**  
  Shows the size of the data at a glance.  
  _Example:_ **50,000 customers** in the dataset.

- **Churn Rate:**  
  Displays the proportion of customers who have churned, or are predicted to churn.  
  _Example:_ **28.9% churn rate**.

- **Average Lifetime Value:**  
  Shows the average value (revenue or profit) each customer brings over their engagement period.  
  _Example:_ **$1,441** average LTV.

### Data Preview Table

- **Purpose:**  
  Gives you a quick look at actual customer records and the range of values for each feature or column.

- **Visible Features Include:**  
  - **Demographics:** Age, Gender, Country, City
  - **Engagement:** Membership_Years, Login_Frequency, Session_Duration_Avg, Pages_Per_Session, Cart_Abandonment_Rate, Wishlist_Items
  - **Purchases:** Total_Purchases, Average_Order_Value, Days_Since_Last_Purchase, Discount_Usage_Rate, Returns_Rate
  - **Communication & Support:** Email_Open_Rate, Customer_Service_Calls, Product_Reviews_Written, Social_Media_Engagement_Score, Mobile_App_Usage
  - **Financial:** Payment_Method_Diversity, Lifetime_Value, Credit_Balance
  - **Churn Data:** Churned
  - **Signup Information:** Signup_Quarter

- **Why it Matters:**  
  - See typical data examples immediately for context
  - Validate data entry, spot weird or missing values
  - Understand what kind of information is being collected and modeled

---

## 2. Feature Distributions

### Dynamic Feature Selector

- Use the dropdown menu to pick any feature (column) and instantly view its distribution.
- Example from the screenshot:  
  **Age Distribution by Churn** – a stacked histogram showing how age values break down, with churned (1) and retained (0) customer counts clearly visible.

### Insights You Can Get

- **Spot patterns or skew:** Are certain age groups more likely to churn?
- **Detect anomalies or outliers:** Are there unusual values (e.g., ages above 100)?
- **Churn risk by segment:** Visually explore how feature values relate to churn.

### Features Available for Analysis

- Full list includes all demographics, behavioral, transactional, and customer value measures mentioned in the preview table.

---

## 3. Missing Value Heatmap

### What You See

- A large heatmap visualization where:
  - **Each column:** Represents a feature.
  - **Each row:** Represents a customer.
  - **Blue marks:** Indicate missing data points.

### How To Use It

- **Quickly identify problematic columns:** Are certain features (e.g., Wishlist_Items, Discount_Usage_Rate) missing lots of data?
- **Plan data cleaning:** Decide if you need to fill, drop, or specially handle features with large gaps.
- **Transparency and confidence:** Quickly ensure you’re not building models on incomplete data.

---

## 4. How to Analyze and Use the EDA Page

- **Validate your data:** Ensure what you have is accurate, consistent, and usable for analysis.
- **Spot trends and risks early:** Understand what features drive churn and where the main gaps or strengths are in your data.
- **Drive your ML workflow:** EDA informs your preprocessing, feature engineering, and the interpretation of results on later pages.

---

## Summary

- The EDA page is your dashboard for dataset health and initial understanding.
- Start here to get instant insight into your customer base, feature quality, and data challenges before heading to segmentation, prediction, or optimization.
- Use the **data overview**, **feature distribution plots**, and **missing value heatmap** together for a full picture of your dataset.

**A strong EDA foundation leads to better models, smarter analytics, and more confident business decisions.**
