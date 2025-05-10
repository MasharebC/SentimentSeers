from app import process_amazon
from yelp_program import process_yelp
from twitter_program import process_twitter
import os

def main():
    print("\n📊 Review Analysis System")
    print("===========================")
    print("1 - Amazon Reviews")
    print("2 - Netflix Reviews")
    print("3 - Twitter Posts")

    choice = input("\nSelect dataset type (1/2/3): ").strip()
    file_path = input("Enter path to the dataset CSV file: ").strip()

    if not os.path.exists(file_path):
        print(f"❌ File not found at: {file_path}")
        return

    if choice == "1":
        process_amazon(file_path)
    elif choice == "2":
        process_yelp(file_path)
    elif choice == "3":
        process_twitter(file_path)
    else:
        print("❌ Invalid option. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
