import pandas as pd
import torch
import matplotlib.pyplot as plt
from evaluation_methods.init_module import *

def get_early_lecture_and_under_name(course):
    course_config = {
        "A-2022": (4, "At_risk_under_D"),
        "B-2020": (4, "At_risk_under_C"),
        "C-2022-1": (8, "At_risk_under_C"),
        "D-2022": (8, "At_risk_under_F"),
        "E-2021": (8, "At_risk_under_F"),
    }
    return next((config for key, config in course_config.items() if key in course), (None, None))

def get_at_risk_number(course, at_risk_category):
    at_risk_dict = {
        "At_risk_under_F": {"A-2022": 2, "B-2020": 4, "C-2022-1": 4, "D-2022": 17, "E-2021": 26},
        "At_risk_under_D": {"A-2022": 24, "B-2020": 6, "C-2022-1": 8, "D-2022": 25, "E-2021": 30},
        "At_risk_under_C": {"A-2022": 29, "B-2020": 18, "C-2022-1": 42, "D-2022": 33, "E-2021": 38},
    }
    return at_risk_dict.get(at_risk_category, {}).get(course, 0)

def analyze_course(course, reg_df, first, last, at_risk_category):
    at_risk = get_at_risk_number(course, at_risk_category)
    
    for lecture in range(first, last + 1):
        df = reg_df.copy()
        model_path = f"./pth_register/best_model_proposed_method_{lecture}.pth"
        model = MultipleLinearRegressionModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        feature_columns = [str(i) for i in range(100)]
        X_pred = df[feature_columns].apply(pd.to_numeric, errors='coerce').values
        df['Predicted_Value'] = [predict(model, x) for x in X_pred]
        
        grouped_predicted_value = df.groupby('user_2')['Predicted_Value'].sum().reset_index()
        grouped_grade = df.groupby('user_2')['grade'].sum().reset_index()
        sorted_grouped_grade = grouped_grade.sort_values(by='grade').head(at_risk)

        num_common_userids = len(
            pd.merge(
                grouped_predicted_value.sort_values(by='Predicted_Value').head(at_risk)['user_2'],
                sorted_grouped_grade['user_2'],
                how='inner', on='user_2'
            )
        )
        
        grade_data_path = f"./Dataset/Course_{course}_GradePoint.csv"
        grade_data = pd.read_csv(grade_data_path)
        merged_df = pd.merge(
            grouped_predicted_value.rename(columns={'user_2': 'userid'}),
            grade_data[['userid', 'grade']],
            on='userid'
        )
        
        grade_order = ['F', 'D', 'C', 'B', 'A']
        merged_df['grade_category'] = pd.Categorical(merged_df['grade'], categories=grade_order, ordered=True)
        merged_df['rank_from_bottom'] = merged_df['Predicted_Value'].rank(ascending=True, method='first').astype(int)

        x_values = merged_df['grade_category'].cat.codes
        y_values = merged_df['rank_from_bottom']
        colors = ['red' if rank <= at_risk else 'blue' for rank in y_values]

        plt.figure(figsize=(7, 4))
        plt.scatter(x_values, y_values, alpha=0.7, c=colors)
        plt.xticks(ticks=range(len(grade_order)), labels=grade_order, fontsize=10)
        plt.xlabel("Grade", fontsize=10)
        plt.ylabel("Rank", fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig(f'./result/{course}_grades_rankings_result.pdf')

def main():
    courses = ["A-2022", "B-2020", "C-2022-1", "D-2022", "E-2021"]
    for course in courses:
        early_lecture, under_name = get_early_lecture_and_under_name(course)
        file_path = f"./learnfd_data/Course{course}_{early_lecture}_100dim.csv"
        reg_df, _ = make_difference_data(file_path, sep=0)
        analyze_course(course, reg_df, first=1, last=1, at_risk_category=under_name)

if __name__ == "__main__":
    main()
