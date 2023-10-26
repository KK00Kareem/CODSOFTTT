import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Data Science\CodSoft Tasks\Task4 Sales Prediction\adv.csv")
df.head()
df.shape
df.describe()

sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.show()

df['TV'].plot.hist(bins=10)
df['Radio'].plot.hist(bins=10, color="red", xlabel="Radio")
df['Newspaper'].plot.hist(bins=10, color="blue", xlabel="Newspaper")

sns.heatmap(df.corr(), annot = True)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3, random_state = 0)

print(X_train)
print(y_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

result = model.predict(X_test)
print(result)
print(y_test)

model.coef_
model.intercept_

0.05473199* 69.2 + 7.14382225

plt.plot(result)

plt.scatter(X_test, y_test)
plt.plot(X_test, 7.14382225 + 0.05473199 * X_test, 'r')
plt.show()