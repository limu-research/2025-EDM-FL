import pandas as pd
import matplotlib.pyplot as plt
from evaluation_methods.init_module import *
from evaluation_methods.evaluate_proposed_method import evaluate_proposed_method
from evaluation_methods.evaluate_random import evaluate_random
from evaluation_methods.evaluate_not_federated import evaluate_not_federated
from evaluation_methods.evaluate_not_difference import evaluate_not_difference

def get_num_lecture_and_under_name(course):
    course_config = {
        "A-2022": (8, "At_risk_under_D"),
        "B-2020": (7, "At_risk_under_C"),
        "C-2022-1": (15, "At_risk_under_C"),
        "D-2022": (16, "At_risk_under_F"),
        "E-2021": (16, "At_risk_under_F"),
    }
    return next((config for key, config in course_config.items() if key in course), (None, None))

def evaluate_methods(course, num_lecture, under_name):
    results = {"proposed": [], "not_federated": [], "not_difference": [], "random": []}
    
    for lecture in range(1, num_lecture + 1):
        file_path = f"./learnfd_data/Course{course}_{lecture}_100dim.csv"
        not_d_df = pd.read_csv(file_path)
        reg_df, _ = make_difference_data(file_path, sep=0)
        method = "FedAvg_E2Vec"
        
        metrics = {
            "proposed": evaluate_proposed_method(course, reg_df, method, first=1, last=10, under_name=under_name),
            "not_federated": evaluate_not_federated(course, reg_df, method, first=1, last=10, under_name=under_name),
            "not_difference": evaluate_not_difference(course, not_d_df, method, first=1, last=10, under_name=under_name),
            "random": evaluate_random(course, reg_df, first=1, last=10, under_name=under_name),
        }
        
        for key, values in metrics.items():
            _, _, _, top_n_precision, ndcg, auc = values
            results[key].append({"lecture": lecture, "at_risk_rate": sum(top_n_precision) / len(top_n_precision)})
            results[key].append({"lecture": lecture, "at_risk_rate": sum(ndcg) / len(ndcg)})
            results[key].append({"lecture": lecture, "at_risk_rate": sum(auc) / len(auc)})
    
    return results

def plot_results(course, results):
    metrics = ["top_n_precision", "ndcg", "PR-AUC"]
    
    for metric, index in zip(metrics, range(0, len(results["proposed"]) // 3)):
        lectures = [item['lecture'] for item in results["proposed"][index::3]]
        
        plt.figure(figsize=(7, 4))
        plt.plot(lectures, [item['at_risk_rate'] for item in results["proposed"][index::3]], marker='o', linestyle='-', color='b', label='Proposed method', zorder=4)
        plt.plot(lectures, [item['at_risk_rate'] for item in results["not_federated"][index::3]], marker='s', linestyle='--', color='y', label='Centralized + Difference Features', zorder=3)
        plt.plot(lectures, [item['at_risk_rate'] for item in results["not_difference"][index::3]], marker='*', linestyle='--', color='g', label='Federated + No Difference Features', zorder=2)
        plt.plot(lectures, [item['at_risk_rate'] for item in results["random"][index::3]], marker='x', linestyle='--', color='r', label='Random', zorder=1)
        
        plt.xlabel('Lecture', fontsize=10)
        plt.ylabel(f'{metric}', fontsize=10)
        plt.xticks(lectures)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        file_name = f'./result/{course}_lecture_sessions_{metric}.pdf'
        plt.savefig(file_name)
        plt.close()

def main():
    courses = ["A-2022", "B-2020", "C-2022-1", "D-2022", "E-2021"]
    
    for course in courses:
        num_lecture, under_name = get_num_lecture_and_under_name(course)
        results = evaluate_methods(course, num_lecture, under_name)
        plot_results(course, results)

if __name__ == "__main__":
    main()
