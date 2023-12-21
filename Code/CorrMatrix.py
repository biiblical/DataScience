import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df = pd.read_csv('C:\\data\\manufacturing_Task_01.csv')
df = pd.DataFrame(df)
corrM = df.corr()
corrM = pd.DataFrame(corrM)
print(corrM)

sb.heatmap(corrM, cmap="Blues", annot=False)
print(corrM.reflectionScore)
