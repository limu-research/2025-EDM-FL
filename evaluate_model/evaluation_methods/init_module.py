import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import auc
import matplotlib.pyplot as plt

class MultipleLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultipleLinearRegressionModel, self).__init__()
        self.hidden1 = nn.Linear(100, 50)
        self.dropout1 = nn.Dropout(p=0.2)
        self.hidden2 = nn.Linear(50, 10)
        self.dropout2 = nn.Dropout(p=0.2)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))   
        x = self.dropout1(x)         
        x = F.relu(self.hidden2(x))   
        x = self.dropout2(x)          
        x = self.output(x)            
        return x

def make_difference_data(file_path, sep=1):
    df = pd.read_csv(file_path)
    class_counts = df["grade"].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    df_filtered = df[df["grade"].isin(valid_classes)]

    if sep == 1:
        a_df, b_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered["grade"], random_state=42)
        
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        
        numeric_a_df = a_df.select_dtypes(include=[int, float])
        for i in range(len(numeric_a_df)):
            for j in range(len(numeric_a_df)):
                if i != j:
                    diff = numeric_a_df.iloc[j] - numeric_a_df.iloc[i]
                    diff = pd.concat([diff, pd.Series([a_df["userid"].iloc[i], a_df["userid"].iloc[j]], index=['user_1', 'user_2'])])
                    df2 = pd.concat([df2, diff.to_frame().T], ignore_index=True)

        numeric_b_df = b_df.select_dtypes(include=[int, float])
        for i in range(len(numeric_b_df)):
            for j in range(len(numeric_b_df)):
                if i != j:
                    diff2 = numeric_b_df.iloc[j] - numeric_b_df.iloc[i]
                    diff2 = pd.concat([diff2, pd.Series([b_df["userid"].iloc[i], b_df["userid"].iloc[j]], index=['user_1', 'user_2'])])
                    df3 = pd.concat([df3, diff2.to_frame().T], ignore_index=True)
        return df2, df3
    else:
        df4 = pd.DataFrame()
        numeric_df = df.select_dtypes(include=[int, float])
        for i in range(len(numeric_df)):
            print(i/len(numeric_df)*100,"%")
            for j in range(len(numeric_df)):
                if i != j:
                    diff4 = numeric_df.iloc[j] - numeric_df.iloc[i]
                    diff4 = pd.concat([diff4, pd.Series([df["userid"].iloc[i], df["userid"].iloc[j]], index=['user_1', 'user_2'])])
                    df4 = pd.concat([df4, diff4.to_frame().T], ignore_index=True)
        return df4,{}

def predict(models, data):
    models.eval() 
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0) 
        predictions = models(data_tensor)
    return predictions.item()
