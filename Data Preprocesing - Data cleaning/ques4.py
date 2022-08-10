import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("pima-indians-diabetes.csv")

data.hist(column="pregs", color="red", edgecolor="green")
plt.xlabel("number of times pregnent")
plt.ylabel("number of patints")
plt.show()

data.hist(column="skin", color="blue", edgecolor="green")
plt.xlabel("Triceps skin fold thickness(mm)")
plt.ylabel("number of patints")
plt.show()
