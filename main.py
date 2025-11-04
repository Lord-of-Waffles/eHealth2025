import sys
import csv
import os
from datetime import datetime
from modules.get_data import get_data
from modules.get_test_data import get_test_data
from modules.run_models import (
    run_clinical,
    run_ct,
    run_pt,
    run_clinical_knn,
    run_ct_knn,
    run_pt_knn,
    run_clinical_ensemble,
    run_ct_ensemble,
    run_pt_ensemble
)
from modules.analyse_model_metrics import analyse_results 

# Configuration
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "model_results.csv")
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.txt")


def setup_results_directory():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")


def initialize_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'model_type',
                'data_type',
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'include_center_id',
                'notes'
            ])


def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")


def save_results_to_csv(model_type, data_type, metrics, include_center_id=None, notes=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            model_type,
            data_type,
            metrics.get('accuracy', ''),
            metrics.get('precision', ''),
            metrics.get('recall', ''),
            metrics.get('f1_score', ''),
            include_center_id if include_center_id is not None else 'N/A',
            notes
        ])
    log_message(f"Results saved: {model_type} on {data_type} - Accuracy: {metrics.get('accuracy', 'N/A')}")


def run_model_with_logging(run_function, model_type, data_type, *args, **kwargs):
    log_message(f"Starting {model_type} training on {data_type} data...")
    try:
        result = run_function(*args, **kwargs)
        if isinstance(result, dict):
            include_center_id = kwargs.get('include_center_id')
            save_results_to_csv(model_type, data_type, result, include_center_id)
            log_message(f"Completed {model_type} on {data_type}")

            # Automatically analyze metrics after each model run
            log_message(f"Analyzing metrics after {model_type} on {data_type}...")
            analyse_results()
            
            return result
        else:
            log_message(f"Warning: {model_type} on {data_type} did not return metrics")
            return None
    except Exception as e:
        log_message(f"ERROR: {model_type} on {data_type} failed - {str(e)}")
        raise


def print_menu():
    print("\n=== MODEL TRAINING & TESTING MENU ===")
    print("1. Clinical (MLP)")
    print("2. CT (MLP)")
    print("3. PET (MLP)")
    print("4. Clinical (kNN)")
    print("5. CT (kNN)")
    print("6. PET (kNN)")
    print("7. Clinical (Ensemble)")
    print("8. CT (Ensemble)")
    print("9. PET (Ensemble)")
    print("10. Run ALL models (comparison)")
    print("0. Exit")
    print("=======================================")


def run_all_models(df_dict, df_test_dict):
    log_message("="*50)
    log_message("RUNNING ALL MODELS FOR COMPARISON")
    log_message("="*50)
    
    models = [
        (run_clinical, "MLP", "Clinical", df_dict["clinical"], df_test_dict["clinical"]),
        (run_ct, "MLP", "CT", df_dict["ct"], df_test_dict["ct"]),
        (run_pt, "MLP", "PET", df_dict["pt"], df_test_dict["pt"]),
        (run_clinical_knn, "kNN", "Clinical", df_dict["clinical"], df_test_dict["clinical"]),
        (run_ct_knn, "kNN", "CT", df_dict["ct"], df_test_dict["ct"]),
        (run_pt_knn, "kNN", "PET", df_dict["pt"], df_test_dict["pt"]),
        (run_clinical_ensemble, "Ensemble", "Clinical", df_dict["clinical"], df_test_dict["clinical"]),
        (run_ct_ensemble, "Ensemble", "CT", df_dict["ct"], df_test_dict["ct"]),
        (run_pt_ensemble, "Ensemble", "PET", df_dict["pt"], df_test_dict["pt"]),
    ]
    
    results = []
    for run_func, model_type, data_type, df_train, df_test in models:
        try:
            if data_type in ["CT", "PET"]:
                result = run_model_with_logging(run_func, model_type, data_type, df_train, df_test, include_center_id=False)
            else:
                result = run_model_with_logging(run_func, model_type, data_type, df_train, df_test)
            results.append((model_type, data_type, result))
        except Exception as e:
            log_message(f"Skipping {model_type} on {data_type} due to error")
            continue
    
    log_message("="*50)
    log_message("ALL MODELS COMPLETED")
    log_message("="*50)

    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    for model_type, data_type, metrics in results:
        if metrics:
            print(f"{model_type:12} | {data_type:8} | Acc: {metrics.get('accuracy', 'N/A'):.4f} | "
                  f"F1: {metrics.get('f1_score', 'N/A'):.4f}")
    print(f"\nDetailed results saved to: {RESULTS_CSV}")


def main():
    setup_results_directory()
    initialize_csv()
    log_message("Program started")

    log_message("Loading training data...")
    df_dict = get_data()
    log_message("Loading test data...")
    df_test_dict = get_test_data()
    log_message("Data loaded successfully")

    while True:
        print_menu()
        choice = input("Select an option (0–10): ").strip()

        if choice == "0":
            log_message("Program exited by user")
            print("Exiting program. Goodbye!")
            sys.exit(0)

        elif choice == "1":
            run_model_with_logging(run_clinical, "MLP", "Clinical", df_dict["clinical"], df_test_dict["clinical"])

        elif choice == "2":
            run_model_with_logging(run_ct, "MLP", "CT", df_dict["ct"], df_test_dict["ct"], include_center_id=False)

        elif choice == "3":
            run_model_with_logging(run_pt, "MLP", "PET", df_dict["pt"], df_test_dict["pt"], include_center_id=False)

        elif choice == "4":
            run_model_with_logging(run_clinical_knn, "kNN", "Clinical", df_dict["clinical"], df_test_dict["clinical"])

        elif choice == "5":
            run_model_with_logging(run_ct_knn, "kNN", "CT", df_dict["ct"], df_test_dict["ct"], include_center_id=False)

        elif choice == "6":
            run_model_with_logging(run_pt_knn, "kNN", "PET", df_dict["pt"], df_test_dict["pt"], include_center_id=False)

        elif choice == "7":
            run_model_with_logging(run_clinical_ensemble, "Ensemble", "Clinical", df_dict["clinical"], df_test_dict["clinical"])

        elif choice == "8":
            run_model_with_logging(run_ct_ensemble, "Ensemble", "CT", df_dict["ct"], df_test_dict["ct"], include_center_id=False)

        elif choice == "9":
            run_model_with_logging(run_pt_ensemble, "Ensemble", "PET", df_dict["pt"], df_test_dict["pt"], include_center_id=False)

        elif choice == "10":
            run_all_models(df_dict, df_test_dict)
            log_message("Starting analysis of model metrics after running all models...")
            analyse_results()

        else:
            print("Invalid choice. Please select a valid option.")

        print("\n--------------------------------------")
        input("Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
