import argparse
import pickle

def display_pickle_file(file_path):
    """
    Opens a pickle file and displays its contents.
    
    :param file_path: Path to the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Contents of the pickle file:")
            print(data)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="display pickle file contents")
    parser.add_argument("--file_path", help="Enter the path to the pickle file", required=True, type=str)
    args = parser.parse_args()
    file_path = args.file_path
    display_pickle_file(file_path)
