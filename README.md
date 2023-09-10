# Social Network Ads Dataset Analysis and Logistic Regression Model

This README file provides an overview of the code and steps for analyzing the Social Network Ads dataset and building a Logistic Regression model using Python and popular data science libraries such as pandas, matplotlib, seaborn, and scikit-learn.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)

## Introduction<a name="introduction"></a>
In this project, we will analyze the Social Network Ads dataset and build a Logistic Regression model to predict whether a user purchased a product based on their age, gender, and estimated salary. This README file will guide you through the steps involved in the analysis and modeling process.

## Data Overview<a name="data-overview"></a>
The dataset, named "Social_Network_Ads.csv," contains the following columns:
- User ID: Unique identifier for each user
- Gender: Gender of the user (Male or Female)
- Age: Age of the user
- EstimatedSalary: Estimated salary of the user
- Purchased: Binary variable (0 or 1) indicating whether the user purchased the product

Let's start by loading the dataset and taking a quick look at its structure.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Display the first few rows of the dataset
df.head()
```

## Data Preprocessing<a name="data-preprocessing"></a>
Before we can build a machine learning model, we need to preprocess the data. In this step, we'll convert the "Gender" column to numeric values and split the data into training and testing sets.

```python
# Convert "Gender" to numeric values (Male: 1, Female: 0)
df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

# Split the data into features (x) and target (y)
x = df.drop("Purchased", axis=1)
y = df[["Purchased"]]
```

## Exploratory Data Analysis (EDA)<a name="exploratory-data-analysis-eda"></a>
EDA is an essential step to understand the relationships between variables and visualize the data. We'll use seaborn and matplotlib for data visualization.

### Pairplot
To visualize relationships between variables, we'll create a pairplot.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df)
plt.show()
```

### Relationship between User ID and Purchased
We'll use jointplot, histplot, and boxplot to explore the relationship between "User ID" and "Purchased."

```python
# Jointplot
sns.jointplot(data=df, x="User ID", y="Purchased")
plt.show()

# Histplot
sns.histplot(data=df, x="User ID", y="Purchased")
plt.show()

# Boxplot
sns.boxplot(data=df, x="Purchased", y="User ID")
plt.show()
```

### Heatmap of Correlations
We can visualize the correlations between numerical features using a heatmap.

```python
sns.heatmap(df.corr(), annot=True)
plt.show()
```

## Model Training<a name="model-training"></a>
Now, we will train a Logistic Regression model using scikit-learn.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)
```

## Model Evaluation<a name="model-evaluation"></a>
Let's evaluate the model's performance on the training and testing data.

```python
# Calculate model accuracy on the training set
train_accuracy = model.score(x_train, y_train)
print("Training Accuracy:", train_accuracy)

# Calculate model accuracy on the testing set
test_accuracy = model.score(x_test, y_test)
print("Testing Accuracy:", test_accuracy)
```

The model's training accuracy and testing accuracy are printed above.

This README provides an overview of the steps involved in analyzing the Social Network Ads dataset and building a Logistic Regression model. You can further enhance the project by fine-tuning the model, conducting feature engineering, and exploring other machine learning algorithms for better predictions.
