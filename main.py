import sys
from modules.get_data import get_data
from modules.get_test_data import get_test_data
from modules.run_models import (
    run_clinical,
    run_ct,
    run_pt,
    run_clinical_knn,
    run_ct_knn,
    run_pt_knn,
)


def print_menu():
    print("\n=== MODEL TRAINING & TESTING MENU ===")
    print("1. Clinical (MLP)")
    print("2. CT (MLP)")
    print("3. PET (MLP)")
    print("4. Clinical (kNN)")
    print("5. CT (kNN)")
    print("6. PET (kNN)")
    print("0. Exit")
    print("=======================================")


def main():
    df_dict = get_data()
    df_test_dict = get_test_data()

    while True:
        print_menu()
        choice = input("Select an option (0–6): ").strip()

        if choice == "0":
            print("Exiting program. Goodbye!")
            sys.exit(0)

        elif choice == "1":
            run_clinical(df_dict["clinical"], df_test_dict["clinical"])

        elif choice == "2":
            run_ct(df_dict["ct"], df_test_dict["ct"], include_center_id=False)

        elif choice == "3":
            run_pt(df_dict["pt"], df_test_dict["pt"], include_center_id=False)

        elif choice == "4":
            run_clinical_knn(df_dict["clinical"], df_test_dict["clinical"])

        elif choice == "5":
            run_ct_knn(df_dict["ct"], df_test_dict["ct"], include_center_id=False)

        elif choice == "6":
            run_pt_knn(df_dict["pt"], df_test_dict["pt"], include_center_id=False)

        else:
            print("Invalid choice. Please select a valid option.")

        print("\n--------------------------------------")
        input("Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
