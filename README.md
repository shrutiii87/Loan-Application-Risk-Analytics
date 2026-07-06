<div>
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020617,10:0F172A,25:1E293B,40:2563EB,55:06B6D4,70:22D3EE,85:A855F7,100:EC4899&height=240&section=header&text=Loan%20Application&fontSize=54&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Turning%20loan%20data%20into%20risk%20insights&descAlignY=60&descSize=20"/>
</div>

Analyzing the **Loan Application** dataset to uncover borrower behavior, income–loan dynamics, and default risk — turning raw loan data into actionable insights with data science.

This project applies **Descriptive Statistics, Probability Theory, Distributions, and Linear Algebra** to a loan applications dataset. It covers central tendency & dispersion, probability and conditional probability, distribution analysis (histograms, skewness, kurtosis, Q-Q plots), and vector-based analysis of borrower income–loan profiles. Using Python, the project highlights how statistical and mathematical concepts are applied in credit risk analysis and Data Science.

✨ *Data-driven insights into loan applications and risk prediction.*

---

## 🎯 Objective

The objective of this project is to analyze loan application data to uncover borrower behavior patterns, evaluate income–loan dynamics, and assess default risk using data science techniques. By applying statistical methods and linear algebra, the project aims to provide actionable insights into credit risk, helping financial institutions make informed lending decisions and improve risk management strategies.

---

## 🗂️ Project Files

| File                                  | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------- |
| 📓 `Loan_application.ipynb`           | Complete data science analysis and Python implementation            |
| 📊 `Loan_applications.csv`            | Loan application dataset (borrower income, loan, credit, default)   |
| 📄 `Loan_Application_theory.pdf`      | Theory, mathematical derivations, formulas, and interpretations     |
| 🖼️ `Diagrams/`                        | Charts, diagrams, and visualizations of borrower behavior           |
| 📘 `README.md`                        | Project documentation                                                |

---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-2C2D72?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Probability-6A1B9A?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Distributions-0EA5E9?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Linear_Algebra-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Credit_Risk_Analysis-059669?style=for-the-badge"/>

</div>

---

# 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1__WPrCETYab-M1IWBLWkbHsIcmElHnY4/view?usp=sharing)

📹 Click the badge above to watch the complete project demonstration.

---

## 📗 Statistical & Mathematical Topics Covered

---

#### 🛠️ Imported Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy import stats
```

---

## 📁 Data Loading & Exploration

```python
df = pd.read_csv('Loan_applications.csv')
```

- `df.head()` / `df.tail()` — preview the first and last 5 rows.
- `df.describe()` — summary statistics of all numeric columns.
- `df.info()` — data types and non-null counts.
- `df.shape` — dataset dimensions.
- `df.columns` — list of all column names.

#### 🎯 Insight:
- The dataset contains borrower-level records with fields such as **Income**, **Loan_Amount**, **Credit_Score**, and **Default_Status** — forming the base for every analysis that follows.

---

## 📊 Step 1: Central Tendency & Dispersion

### 🔢 (a) Mean, median, and mode of Income

```python
mean_income = df['Income'].mean()
median_income = df['Income'].median()
mode_income = df['Income'].mode()
```

#### 🎯 Insight:
- **Mean Income** 📊 → ₹109,639 → Slightly higher than the median, hinting at some high earners pulling the average up.
- **Median Income** 📈 → ₹108,503 → Close to the mean, suggesting a fairly balanced distribution.
- **Mode Income** 🔄 → ₹43,670 → Much lower than mean/median, showing a cluster of borrowers earning in this band.

---

### 🔢 (b) Range, variance, and standard deviation of Loan_Amount

```python
loan_range = df['Loan_Amount'].max() - df['Loan_Amount'].min()
loan_variance = df['Loan_Amount'].var()
loan_std = df['Loan_Amount'].std()
```

#### 🎯 Insight:
- **Range** 📏 → 494,647 → Loans vary widely across borrowers.
- **Variance** 📐 → 20.4B → Huge spread in loan values.
- **Std Dev** 📊 → 142,845 → Average loan deviation is very high.
- ✨ Loan amounts are highly inconsistent — a big gap exists between small and large loans.

---

## 📉 Step 2: Probability & Events

### 🔢 (a) Probability of loan default

```python
default_count = (df['Default_Status(Yes/No)'] == 'Yes').sum()
total_customers = len(df)
prob_default = default_count / total_customers
```

#### 🎯 Insight:
- **Default Probability** 📊 → 0.5602 (≈56%) → More than half of customers are likely to default.
- **Risk Signal** ⚠️ → High default rate indicates significant credit risk.
- **Business Impact** 💰 → Lenders must tighten screening or risk heavy losses.

---

### 🔢 (b) Contingency table: Default_Status vs Credit_Score category

```python
df['Credit_Category'] = pd.cut(
    df['Credit_Score'],
    bins=[300, 600, 700, 850],
    labels=['Low', 'Medium', 'High'])

contingency_table = pd.crosstab(
    df['Credit_Category'],
    df['Default_Status(Yes/No)'])
```

#### 🎯 Insight:
- **Low Credit** ⚠️ → 2,152 defaults vs 576 non-defaults → Very high-risk group.
- **Medium Credit** 📉 → 255 defaults vs 658 non-defaults → Moderate risk, better balance.
- **High Credit** ✅ → 384 defaults vs 962 non-defaults → Safest group, majority repay.
- ✨ Default risk drops sharply as credit score improves — strong evidence that credit score is a reliable predictor of repayment behavior.

---

### 🔢 (c) Conditional probability: P(Default | Credit_Score < 600)

```python
low_credit = df[df['Credit_Score'] < 600]
conditional_probability = (low_credit['Default_Status(Yes/No)'] == 'Yes').sum() / len(low_credit)
```

#### 🎯 Insight:
- **Conditional Probability** 📊 → 0.79 (≈79%) → Customers with low credit scores (<600) have a very high chance of defaulting.
- **Business Impact** 💰 → Strong need for stricter lending policies for this segment.
- ✨ Low credit score is a powerful predictor of default — lenders should treat this segment as high-risk.

---

## 📐 Step 3: Distributions & Visualization

### 🔢 (a) Histogram of Credit_Score with a Gaussian curve

```python
plt.hist(df['Credit_Score'], bins=20, density=True, alpha=0.6, color='#9932CC', edgecolor='black')

mean = df['Credit_Score'].mean()
std = df['Credit_Score'].std()
x = np.linspace(df['Credit_Score'].min(), df['Credit_Score'].max(), 100)

plt.plot(x, norm.pdf(x, mean, std))
```

#### 🎯 Insight:
- **Distribution Shape** 📈 → Roughly uniform with a slight peak near mid-range (≈600).
- **Spread** 📏 → Scores span 300–850, showing wide variation among customers.
- ✨ Credit scores are fairly balanced — not heavily skewed, suggesting a diverse mix of borrower profiles.

---

### 🔢 (b) Skewness and Kurtosis of Loan_Amount

```python
loan_skewness = skew(df['Loan_Amount'])
loan_kurtosis = kurtosis(df['Loan_Amount'])
```

#### 🎯 Insight:
- **Skewness** ⚖️ → 0.008 → Almost zero, meaning the loan distribution is nearly symmetric.
- **Kurtosis** 📉 → −1.20 → Indicates a flat (platykurtic) shape — fewer extreme loan values than a normal curve.
- ✨ Loan amounts are evenly spread with minimal skew and light tails, suggesting stable, moderate variation among borrowers.

---

### 🔢 (c) Q-Q plot for Income

```python
stats.probplot(df['Income'], dist="norm", plot=plt)
```

#### 🎯 Insight:
- **Normality Check** 📈 → Points deviate from the reference line at both ends, showing the data isn't perfectly normal.
- **Tail Behavior** 💰 → Curved tails indicate right skew — a few high-income outliers.
- ✨ Income distribution is moderately skewed — a transformation (like Box-Cox or log) could help normalize it for better statistical modeling.

---

## 🧾 Step 4: Linear Algebra Application

### 🔷 Represent each customer's [Income, Loan_Amount] as a vector

```python
vectors = df[['Income', 'Loan_Amount']].head(5)
v1 = vectors.iloc[0].values
v2 = vectors.iloc[1].values
```

### 🔢 (a) Dot product between two customer vectors

```python
dot_product = np.dot(v1, v2)
```

#### 🎯 Insight:
- **Dot Product** 📊 → 8.82B → A very large value, showing strong alignment between the two vectors.
- ✨ Both borrowers have aligned income–loan profiles, reinforcing the correlation between earnings and loan size.

---

### 🔢 (b) L2 Norm of a customer's financial vector

```python
norm = np.linalg.norm(v1)
```

#### 🎯 Insight:
- **L2 Norm** 📏 → 52,413 → Represents the overall "magnitude" of the borrower's vector (Income + Loan).
- ✨ The norm acts like the "length" of the financial vector — larger values mean a higher combined scale of income and loan.

---

### 🔢 (c) Angle between two customer vectors

```python
cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.degrees(np.arccos(cos_theta))
```

#### 🎯 Insight:
- **Angle** 📐 → 62.36° → The vectors are not perfectly aligned, showing a noticeable difference in borrower profiles.
- **Cosine Similarity** 🔗 → Moderate — they share some proportionality but diverge significantly.
- ✨ These two borrowers have related but distinct income–loan patterns, not pointing in the same financial direction.

---

## 🏁 Overall Conclusion & Key Takeaways

#### 📌 Dataset Summary:
- Loan applicant records with Income, Loan_Amount, Credit_Score, and Default_Status ✅
- Wide variation in loan amounts and a high overall default rate (~56%)

#### 🔗 Statistical Findings:
- Income and Loan_Amount are moderately spread, with income showing a right-skewed tail of high earners.
- Loan amounts are nearly symmetric (low skew) but highly variable (large std. deviation).

#### 🌟 Probability & Risk Insights:
- Credit score is a strong predictor of default: low scores (<600) carry a ≈79% conditional default probability.
- The contingency table confirms default rates fall sharply as credit category improves.

#### 🧭 Linear Algebra Insights:
- Representing borrowers as income–loan vectors reveals both similarity (via dot product & norm) and divergence (via angle) between financial profiles.
- These geometric measures offer an alternative, mathematically grounded way to compare borrower profiles beyond raw statistics.

#### ✅ Final Verdict:
- Credit score emerges as the strongest signal for default risk in this dataset, while income and loan amount show independent but informative variation. Combining statistical and vector-based analysis gives lenders a well-rounded view of borrower risk. 🚀

---

# 🚀 How to Run

Install the required libraries:

```bash
pip install pandas numpy scipy matplotlib
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebook:

```text
Loan_application.ipynb
```

---

## 👩‍💻 Shruti Bhawsar

📍 Ahmedabad |

[![GitHub](https://img.shields.io/badge/GitHub-shrutiii87-181717?style=flat&logo=github)](https://github.com/shrutiii87)

---

⭐ **If You Like This Project**

Give this repository a ⭐ and feel free to fork or contribute!

📊 *Clean Models · Sharp Insights · Confident Conclusions*
