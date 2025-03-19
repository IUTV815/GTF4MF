import pandas as pd
import numpy as np

input_file = "./dataset/log_Female_Mortality.csv"
output_file = "./dataset/log_log_Female_Mortality.csv"

df = pd.read_csv(input_file)
columns_to_transform = df.columns.difference(["Age", "Year"])
df[columns_to_transform] = np.log(df[columns_to_transform])
df.to_csv(output_file, index=False)

print(f"已将对数转换后的log_Female_Mortality数据保存至 {output_file}")

input_file = "./dataset/log_Male_Mortality.csv"
output_file = "./dataset/log_log_Male_Mortality.csv"

df = pd.read_csv(input_file)
columns_to_transform = df.columns.difference(["Age", "Year"])
df[columns_to_transform] = np.log(df[columns_to_transform])
df.to_csv(output_file, index=False)

print(f"已将对数转换后的log_Male_Mortality数据保存至 {output_file}")
