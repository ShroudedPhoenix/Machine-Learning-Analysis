import data_import
import machine_learning
import predictions


def main():
    # Set the test and data folders
    test_folder = r'test'
    data_folder = r'data'

    data_import.combine_data()
    # machine_learning.ML_run()
    predictions.run_predictions()

    print("Process complete.")


if __name__ == "__main__":
    main()
