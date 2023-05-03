import pandas as pd

data = pd.read_csv("dataset.csv")
label_map ={ "low risk":1 ,"mid risk":2, "high risk":3}
data["RiskLevel"]= data["RiskLevel"].map(label_map)
data.to_csv("dataset.csv", index=False)