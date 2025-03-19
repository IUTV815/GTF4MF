import os
import json
import csv

# 文件前缀和后缀
base_dir = "/data/gqyu/New/lpm/LPM/test_dict/Male/MM/"
models = ["Autoformer", "Crossformer", "iTransformer", "Leddam",'PatchTST','SOFTS','Transformer','TSMixer']  # 假设你知道有哪些model
seq_lens = [100, 200, 300, 400]  # 假设你知道有哪些seqlen
pred_lens = [100, 200, 300, 400,1]  # 假设你知道有哪些predlen

# 提取test_dict后续两个文件夹的名字（例如 "Female" 和 "MM"）
folders = base_dir.rstrip(os.sep).split(os.sep)
output_csv_name = f"{folders[-2]}_{folders[-1]}"  # 组合成 "Female_MM"
# 确保文件夹 './result' 存在，不存在时自动创建
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)
# 定义输出的csv文件路径
output_csv = os.path.join(output_dir, f"{output_csv_name}.csv")

# 初始化一个列表，用来存储csv的数据
rows = []

# 遍历所有可能的model, seqlen, predlen
for model in models:
    for seq_len in seq_lens:
        for pred_len in pred_lens:
            # 构造文件路径
            file_path = os.path.join(base_dir, model, f"{seq_len}_{pred_len}", "records.json")
            
            if os.path.exists(file_path):  # 确保文件存在
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        
                        # 提取mse和mae
                        mse = data.get("mse", None)
                        mae = data.get("mae", None)
                        
                        if mse is not None and mae is not None:
                            # 将信息添加到行数据中
                            rows.append([model, seq_len, pred_len, mse, mae])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# 如果rows列表不为空，则将数据写入csv文件
if rows:
    # 打开csv文件并写入数据
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "seqlen", "predlen", "mse", "mae"])  # 写入标题行
        writer.writerows(rows)  # 写入数据行
    print(f"Data successfully written to {output_csv}")
else:
    print("No data found to write to CSV.")
