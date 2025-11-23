# Cost-Sensitive Optimization Page

This page of the app helps you use your churn predictions to make smarter, money-saving business decisions—by weighing the **real cost** of customer churn and retention campaigns.

---

## What Does the "Cost-Sensitive Optimization" Page Do?

Predicting who is likely to leave is only half the story. Deciding **how many customers to target for retention (and at what cutoff)** is just as important. This page guides you in finding the most cost-effective way to act on churn predictions.

---

## Main Features

### 1. **Cost Inputs**
- **Cost of False Negative ($):**  
  Enter the estimated loss if you fail to save a customer who actually leaves (e.g., lost revenue per churned customer).
- **Cost of False Positive ($):**  
  Enter the cost if you act (offer a retention deal) to a customer who would have stayed anyway (e.g., cost of a coupon or special marketing).

### 2. **Expected Loss vs Threshold Graph**
- This line chart shows how your **expected total cost** (from both missed churners and “unnecessary” retention offers) changes as you adjust the *churn probability threshold*.
- **Threshold** = How aggressive you are in flagging customers for retention:
  - Lower threshold = More customers get offers (more prevention, higher cost but fewer lost customers).
  - Higher threshold = Fewer offers (lower cost, more missed churners, higher revenue loss).
- **Dashed Red Line:** Marks the point on the curve where your expected cost is minimized—the “sweet spot.”

### 3. **Optimal Threshold & Minimum Expected Cost**
- **Optimal Threshold:**  
  The ideal cutoff for the churn probability to minimize total loss, given your chosen costs.
- **Minimum Expected Cost:**  
  The lowest total loss you can expect by following the optimal threshold strategy.

---

## How To Use This Page

1. **Set your business-specific costs** in the two boxes at the top (false negatives & false positives).
2. **Read the graph:**  
   - The curve tells you how your loss changes with the threshold.
   - The lowest point (red line) = where your retention strategy is most efficient.
3. **Check the recommended threshold** and **minimum cost** shown below the chart.
4. **Apply this threshold** in your campaign targeting, using it as the cutoff probability for selecting which customers should receive retention offers/interventions.

---

## Why Is This Important?

- **No guesswork:** Move beyond using a “default” 0.5 cutoff, and set a threshold tailored to your ROI and costs.
- **Perfect balance:** Save retention dollars by not overspending, but don’t lose revenue by missing at-risk customers.
- **Business-tailored:** Whether your offers are expensive or cheap, and your lost customer value high or low, the tool adapts to your needs.

---

## Summary

This page turns machine learning churn probabilities into real business recommendations, guiding you to maximize profit and minimize loss. Enter your costs, analyze the graph, and confidently decide how to act on churn predictions.

**In short: Use this page to make your churn model work for your bottom line!**
