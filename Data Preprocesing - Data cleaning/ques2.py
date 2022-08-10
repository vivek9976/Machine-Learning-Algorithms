import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("pima-indians-diabetes.csv")
# between age and pregs
plt.scatter(data["Age"], data["pregs"])
plt.title("Age and pregs")
plt.show()

# between age and plas
plt.scatter(data["Age"], data["plas"])
plt.title("Age and plas")
plt.show()
# between age and pres
plt.scatter(data["Age"], data["pres"])
plt.title("Age and pres")
plt.show()
# between age and skin
plt.scatter(data["Age"], data["skin"])
plt.title("Age and skin")
plt.show()
# between age and BMI
plt.scatter(data["Age"], data["BMI"])
plt.title("Age and BMI")
plt.show()
# between age and test
plt.scatter(data["Age"], data["test"])
plt.title("Age and test")
plt.show()
# between age and pedi
plt.scatter(data["Age"], data["pedi"])
plt.title("Age and pedi")
plt.show()
# between BMI and pregs
plt.scatter(data["BMI"], data["pregs"])
plt.title("BMI and pregs")
plt.show()
# between BMI and plas
plt.scatter(data["BMI"], data["plas"])
plt.title("BMI and plas")
plt.show()
# between BMI and pres
plt.scatter(data["BMI"], data["pres"])
plt.title("BMI and pres")
plt.show()
# between BMI and skin
plt.scatter(data["BMI"], data["skin"])
plt.title("BMI and skin")
plt.show()
# between BMI and Age
plt.scatter(data["BMI"], data["Age"])
plt.title("BMI and Age")
plt.show()
# between BMI and test
plt.scatter(data["BMI"], data["test"])
plt.title("BMI and test")
plt.show()
# between BMI and pedi
plt.scatter(data["BMI"], data["pedi"])
plt.title("BMI and pedi")
plt.show()
