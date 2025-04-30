from .init_module import *

def load_model(method, number):
    pth = fr"best_model_proposed_method_{number}.pth"
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
    sorted_by_predicted = df.groupby('user_2')['Predicted_Value'].sum().reset_index().sort_values(by='Predicted_Value').head(num)
    sorted_by_grade = df.groupby('user_2')['grade'].sum().reset_index().sort_values(by='grade').head(at_risk)
    common_users = pd.merge(sorted_by_predicted[['user_2']], sorted_by_grade[['user_2']], on='user_2', how='inner')
    return len(common_users) / num

def calculate_ndcg(df, df2, at_risk):
    sorted_grouped_predicted_value = df.groupby('user_2')['Predicted_Value'].sum().reset_index().sort_values(by='Predicted_Value')
    df2 = df2.rename(columns={'userid': 'user_2'})  # `userid` を `user_2` に統一
    merged_df = pd.merge(sorted_grouped_predicted_value, df2[['user_2', 'grade']], on='user_2', how='left')
    merged_df['grade'] = 1 - merged_df['grade']

    dcg = sum(merged_df["grade"].iloc[i] / np.log2(i + 1) if i > 0 else merged_df["grade"].iloc[i] for i in range(at_risk))
    
    sorted_df_grade = df2.sort_values(by='grade', ascending=True)
    sorted_df_grade['grade'] = 1 - sorted_df_grade['grade']
    dcg_perfect = sum(sorted_df_grade["grade"].iloc[i] / np.log2(i + 1) if i > 0 else sorted_df_grade["grade"].iloc[i] for i in range(at_risk))

    return dcg / dcg_perfect if dcg_perfect > 0 else 0

def calculate_auc(df, at_risk):
    data = [{'i': 0, 'recall': 0, 'precision': 1}]

    grouped_predicted_value = df.groupby('user_2')['Predicted_Value'].sum().reset_index()
    grouped_grade = df.groupby('user_2')['grade'].sum().reset_index()

    for i in range(1, len(grouped_predicted_value)):
        sorted_grouped_predicted_value = grouped_predicted_value.sort_values(by='Predicted_Value').head(i)
        sorted_grouped_grade = grouped_grade.sort_values(by='grade').head(at_risk)
        common_users = pd.merge(sorted_grouped_predicted_value[['user_2']], sorted_grouped_grade[['user_2']], on='user_2', how='inner')

        recall = len(common_users) / at_risk
        precision = len(common_users) / i
        data.append({'i': i, 'recall': recall, 'precision': precision})

    df_auc = pd.DataFrame(data)
    return auc(df_auc['recall'], df_auc['precision'])

def evaluate_proposed_method(Course, reg_df, method, first, last, under_name):
    top_5_precision, top_10_precision, top_15_precision, top_n_precision, ndcg_results, auc_results = [],[],[],[],[],[]
    feature_columns = [str(i) for i in range(100)]
    
    for number in range(first, last + 1):
        df = reg_df.copy()
        model = load_model(method, number)
        df = predict_values(model, df, feature_columns)

        at_risk = get_at_risk_threshold(Course, under_name)
        if at_risk == 0:
            continue

        grades_file_path = fr"./learnfd_data/Course{Course}_100dim.csv"
        df2 = pd.read_csv(grades_file_path)
        top_5_precision.append(calculate_top_n_precision(df, 5, at_risk))
        top_10_precision.append(calculate_top_n_precision(df, 10, at_risk))
        top_15_precision.append(calculate_top_n_precision(df, 15, at_risk))
        top_n_precision.append(calculate_top_n_precision(df, at_risk, at_risk))
        ndcg_results.append(calculate_ndcg(df, df2, at_risk))
        auc_results.append(calculate_auc(df, at_risk))

    return top_5_precision, top_10_precision, top_15_precision, top_n_precision, ndcg_results, auc_results
