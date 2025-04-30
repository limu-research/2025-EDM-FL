from .init_module import *

def load_model(method, number):
    pth = fr"best_model_not_difference_{number}.pth"
    model = MultipleLinearRegressionModel()
    model.load_state_dict(torch.load(fr"./pth_register/{pth}", map_location=torch.device('cpu')))
    return model

def predict_values(model, df, feature_columns):
    X_pred = df[feature_columns].apply(pd.to_numeric, errors='coerce').values
    df['Predicted_Value'] = [predict(model, x) for x in X_pred]
    return df

def get_at_risk_threshold(Course, under_name):
    thresholds = {
        "At_risk_under_F": {"A-2022": 2, "B-2020": 4, "C-2022-1": 4, "D-2022": 17, "E-2021": 26},
        "At_risk_under_D": {"A-2022": 24, "B-2020": 6, "C-2022-1": 8, "D-2022": 25, "E-2021": 30},
        "At_risk_under_C": {"A-2022": 29, "B-2020": 18, "C-2022-1": 42, "D-2022": 33, "E-2021": 38},
    }
    return thresholds.get(under_name, {}).get(Course, 0)

def calculate_top_n_precision(df, num, at_risk):
    sorted_by_predicted = df.sort_values(by='Predicted_Value').head(num)
    sorted_by_grade = df.sort_values(by='grade').head(at_risk)
    common_users = pd.merge(sorted_by_predicted[['userid']], sorted_by_grade[['userid']], on='userid', how='inner')
    return len(common_users) / num

def calculate_ndcg(df, Course, at_risk):
    df = df.sort_values(by='Predicted_Value')
    df['grade'] = 1 - df['grade']

    dcg = sum(df["grade"].iloc[i] / np.log2(i + 1) if i > 0 else df["grade"].iloc[i] for i in range(at_risk))

    df_sorted = df.sort_values(by='grade', ascending=False)
    dcg_perfect = sum(df_sorted["grade"].iloc[i] / np.log2(i + 1) if i > 0 else df_sorted["grade"].iloc[i] for i in range(at_risk))

    return dcg / dcg_perfect if dcg_perfect > 0 else 0

def calculate_auc(df, at_risk):
    data = [{'i': 0, 'recall': 0, 'precision': 1}]

    for i in range(1, len(df)):
        sorted_predicted = df.sort_values(by='Predicted_Value').head(i)
        sorted_grade = df.sort_values(by='grade').head(at_risk)
        common_users = pd.merge(sorted_predicted[['userid']], sorted_grade[['userid']], on='userid', how='inner')

        recall = len(common_users) / at_risk
        precision = len(common_users) / i
        data.append({'i': i, 'recall': recall, 'precision': precision})

    df_auc = pd.DataFrame(data)
    return auc(df_auc['recall'], df_auc['precision'])

def evaluate_not_difference(Course, reg_df, method, first, last, under_name):
    top_5_precision, top_10_precision, top_15_precision, top_n_precision, ndcg_results, auc_results = [],[],[],[],[],[]
    feature_columns = [str(i) for i in range(100)]
    
    for number in range(first, last + 1):
        df = reg_df.copy()
        model = load_model(method, number)
        df = predict_values(model, df, feature_columns)

        at_risk = get_at_risk_threshold(Course, under_name)
        top_5_precision.append(calculate_top_n_precision(df, 5, at_risk))
        top_10_precision.append(calculate_top_n_precision(df, 10, at_risk))
        top_15_precision.append(calculate_top_n_precision(df, 15, at_risk))
        top_n_precision.append(calculate_top_n_precision(df, at_risk, at_risk))
        ndcg_results.append(calculate_ndcg(df, Course, at_risk))
        auc_results.append(calculate_auc(df, at_risk))

    return top_5_precision, top_10_precision, top_15_precision, top_n_precision, ndcg_results, auc_results
