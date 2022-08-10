import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("pima-indians-diabetes.csv")
data.groupby("class").hist(column="pregs", edgecolor="green")
plt.show()
