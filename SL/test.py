import pandas as pd

df = pd.read_csv("C:\\Users\\samth\\source\\repos\\MTG_Draft_Agent\\SL\\data\\raw\\replay_data_public.SOS.PremierDraft.csv", nrows=5)

columns = df.columns.tolist()
for i in range(len(columns)):
    print(columns[i])

print("\nSAMPLE ROWS:")
print(df)

