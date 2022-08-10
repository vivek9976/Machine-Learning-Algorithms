import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("pima-indians-diabetes.csv")

print("correlation b/w Age and pregs",data["pregs"].corr(data["Age"]))
print("correlation b/w Age and plas",data["plas"].corr(data["Age"]))
print("correlation b/w Age and pres",data["pres"].corr(data["Age"]))
print("correlation b/w Age and skin",data["skin"].corr(data["Age"]))
print("correlation b/w Age and BMI",data["BMI"].corr(data["Age"]))
print("correlation b/w Age and test",data["test"].corr(data["Age"]))
print("correlation b/w Age and pedi",data["pedi"].corr(data["Age"]))
print("correlation b/w Age and age",data["Age"].corr(data["Age"]))


print("correlation b/t BMI and pregs",data["pregs"].corr(data["BMI"]))
print("correlation b/t BMI and plas",data["plas"].corr(data["BMI"]))
print("correlation b/t BMI and pres",data["pres"].corr(data["BMI"]))
print("correlation b/t BMI and skib",data["skin"].corr(data["BMI"]))
print("correlation b/t BMI and test",data["test"].corr(data["BMI"]))
print("correlation b/t BMI and padi",data["pedi"].corr(data["BMI"]))
print("correlation b/t BMI and age",data["Age"].corr(data["BMI"]))
print("correlation b/t BMI and bmi",data["BMI"].corr(data["BMI"]))
