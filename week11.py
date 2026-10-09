import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('Iris.csv')
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

print("--- Dataset head ---")
print(df.head())

print("\n--- Dataset info ---")
print(df.info())

#Summary statistics
print("\nSummary Statistics (Overall)")
print(df.describe())

print("\nSummary Statistics (Grouped by Species)")
print(df.groupby('Species').describe().T)

print("Species Class Distribution")
print(df['Species'].value_counts())

#Data Visualization
sns.set_theme(style="whitegrid")

#Plot 1: Feature Distributions (Histograms + KDE)
plt.figure(figsize=(12, 8))
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for i, col in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=col, hue='Species', kde=True, element = 'step')
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

#Plot 2: Boxplots to Detect Outliers and compare Distributions
plt.figure(figsize=(12, 8))
for i, col in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='Species', y=col, palette='Set2')
    plt.title(f'{col} by Species')

plt.tight_layout()
plt.show()

#Plot 3: Pairplot (Feature Interactions Across Species)
sns.pairplot(df, hue='Species', corner=True, palette = 'Dark2')
plt.suptitle('Pairwise Feature Relationships', y = 1.02)
plt.show()

#Plot 4: Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.drop(columns=['Species'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()
