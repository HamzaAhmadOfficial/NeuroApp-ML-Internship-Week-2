import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("titanic_cleaned.csv")
os.makedirs("visualizations", exist_ok=True)

# Line plot
plt.figure()
plt.plot(df['Age'])
plt.savefig("visualizations/line_plot_age.png")
plt.close()

# Scatter plot
plt.figure()
plt.scatter(df['Age'], df['Fare'])
plt.savefig("visualizations/scatter_age_fare.png")
plt.close()

# Histogram
plt.figure()
df['Pclass'].hist()
plt.savefig("visualizations/histogram_pclass.png")
plt.close()

# Bar chart
plt.figure()
sns.barplot(x='Sex', y='Survived', data=df)
plt.savefig("visualizations/bar_chart_sex_survived.png")
plt.close()

# Box plot
plt.figure()
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.savefig("visualizations/box_plot_pclass_fare.png")
plt.close()

# Violin plot
plt.figure()
sns.violinplot(x='Sex', y='Age', data=df)
plt.savefig("visualizations/violin_plot.png")
plt.close()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.savefig("visualizations/heatmap.png")
plt.close()

# Pair plot
pairplot = sns.pairplot(df[['Age', 'Fare', 'Survived', 'FamilySize']])
pairplot.savefig("visualizations/pairplot.png")
