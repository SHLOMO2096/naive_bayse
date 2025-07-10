class Menu:

    def display(self):
        while True:
            print("\n--- Naive Bayes Classifier Menu ---")
            print("1. Load & clean data")
            print("2. Train model")
            print("3. Evaluate model")
            print("4. Classify new record")
            print("5. Exit")

            choice = input("Choose an option (1-5): ")

            if choice == '1':
                target_column = input("Enter target column name: ")
                self.app = App(target_column)
                self.app.load_and_clean()
            elif choice == '2':
                self.app.train_model()
            elif choice == '3':
                self.app.evaluate_model()
            elif choice == '4':
                self.app.classify_record()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Try again.")


if __name__ == "__main__":
    from app import App
    menu = Menu()
    menu.display()
