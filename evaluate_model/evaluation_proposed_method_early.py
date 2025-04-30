import pandas as pd
from evaluation_methods.init_module import *
from evaluation_methods.evaluate_proposed_method import evaluate_proposed_method

def get_early_lecture_and_under_name(course):
    course_config = {
        "A": (4, "At_risk_under_D"),
        "B": (4, "At_risk_under_C"),
        "C": (8, "At_risk_under_C"),
        "D": (8, "At_risk_under_F"),
        "E": (8, "At_risk_under_F"),
    }
    return next((config for key, config in course_config.items() if key in course), (None, None))

def main():
    courses = ["A-2022", "B-2020", "C-2022-1", "D-2022", "E-2021"]
    results = []
    
    for course in courses:
        early_lecture, under_name = get_early_lecture_and_under_name(course)
        file_path = f"./learnfd_data/Course{course}_{early_lecture}_100dim.csv"
        reg_df, _ = make_difference_data(file_path, sep=0)
        method = "FedAvg_E2Vec"
        
        metrics = evaluate_proposed_method(course, reg_df, method, first=1, last=10, under_name=under_name)
        avg_metrics = [sum(metric) / len(metric) for metric in metrics]
        results.append([course] + avg_metrics)
    
    columns = [
        "Course", "Top-n-precision (n=5)", "Top-n-precision (n=10)",
        "Top-n-precision (n=15)", "Top-n-precision (n=At-risk)", "ndcg", "PR-AUC"
    ]
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv("./result/evaluate_proposed_method_early.csv", index=False)
    
    print("Results saved to ./result/evaluate_proposed_method_early.csv")

if __name__ == "__main__":
    main()
