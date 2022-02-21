import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("../0data/A414.csv")
df = df.iloc[:len(df)//100]
print(df.columns)
plt.plot(df["TimePeriod"],df["Flow"])
plt.show()
