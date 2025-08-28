# 📌 Data Cleaning Guide

## 📖 Introduction

**Data Cleaning** is the process of identifying and correcting (or removing) errors, inconsistencies, and inaccuracies in a dataset. Cleaning ensures that data is structured, reliable, and ready for analysis or machine learning models.

### 📌 Why is Data Cleaning Important?

🔹 **Improves Data Accuracy** – Errors and inconsistencies can lead to incorrect conclusions.  
🔹 **Enhances Model Performance** – Clean data leads to better machine learning results.  
🔹 **Eliminates Redundancy** – Duplicate records waste storage and computing resources.  
🔹 **Ensures Better Decision-Making** – Reliable data improves business intelligence.  

---

## 📂 Example Dataset: `student_scores.csv`

Let's consider a dataset containing student scores with common issues such as missing values, duplicate records, and outliers.

| Student_ID | Name   | Age | Score  | Subject  | Gender |
|------------|--------|-----|--------|----------|--------|
| 101        | Prince  | 26  | 85     | Math     | F      |
| 102        | Lovnish    |     | 90     | Science  | M      |
| 103        | Ravi  | 18  | -1     | Math     |        |
| 104        | Pranav   | 19  | 88     | Science  | M      |
| 105        | Chandan    | 17  | 92     | Math     | F      |
| 106        | Rajat | 20  | 89     |          | M      |

### 🛑 Issues in the Dataset:
✅ Missing values in `Age`, `Gender`, and `Subject` columns.  
✅ Invalid values (negative score).  
✅ Inconsistent formatting and duplicate records.  
✅ Outliers in the data.

---

## 🛠 Data Cleaning Techniques

### 1️⃣ Load the Dataset
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('student_scores.csv')
```

### 2️⃣ Handling Missing Values

#### 🔍 Identify Missing Values
```python
print(df.isnull().sum())  # Check missing values in each column
```

#### 🏗 Fill Missing Values
- **For Numerical Data (Age)**: Fill with median value
```python
df['Age'].fillna(df['Age'].median(), inplace=True)
```
- **For Categorical Data (Gender & Subject)**: Fill with mode
```python
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Subject'].fillna('Unknown', inplace=True)
```

### 3️⃣ Handling Invalid Values

#### ❌ Removing Negative Scores
```python
df = df[df['Score'] >= 0]  # Remove rows where Score is negative
```

### 4️⃣ Handling Duplicate Records
```python
df.drop_duplicates(inplace=True)
```

### 5️⃣ Standardizing Text Data
```python
df['Name'] = df['Name'].str.title()  # Capitalize Names
df['Subject'] = df['Subject'].str.capitalize()  # Standardize Subjects
```

### 6️⃣ Handling Outliers

**Using IQR (Interquartile Range) to detect outliers:**
```python
Q1 = df['Score'].quantile(0.25)
Q3 = df['Score'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Score'] >= Q1 - 1.5 * IQR) & (df['Score'] <= Q3 + 1.5 * IQR)]
```

### 7️⃣ Convert Data Types
```python
df['Student_ID'] = df['Student_ID'].astype(str)  # Convert Student_ID to string
df['Age'] = df['Age'].astype(int)  # Convert Age to integer
```

### 8️⃣ Save Cleaned Data
```python
df.to_csv('cleaned_student_scores.csv', index=False)  # Save to a new file
```

---

## ✅ Summary of Cleaning Steps
- **Handled missing values** (numerical → median, categorical → mode)
- **Removed invalid and duplicate records**
- **Standardized text formatting**
- **Detected and removed outliers using IQR**
- **Converted data types correctly**
- **Saved the cleaned data for further analysis**

With this guide, you can efficiently clean and preprocess datasets, ensuring higher accuracy in data analysis and machine learning models. 🚀

