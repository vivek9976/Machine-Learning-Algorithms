import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("pima-indians-diabetes.csv")

plt.boxplot(data["pregs"])
plt.show()
plt.boxplot(data["plas"])
plt.show()
plt.boxplot(data["pres"])
plt.show()
plt.boxplot(data["skin"])
plt.show()
plt.boxplot(data["BMI"])
plt.show()
plt.boxplot(data["test"])
plt.show()
plt.boxplot(data["pedi"])
plt.show()
plt.boxplot(data["Age"])
plt.show()
