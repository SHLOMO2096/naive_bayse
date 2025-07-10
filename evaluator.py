class Evaluator:
    def evaluate_sample(self, classifier, record, true_label):
        predicted = classifier.classify(record)
        if predicted == true_label:
            print(f"Correctly classified: {record} as {predicted}")
            return True
        else:
            print(f"Incorrectly classified: {record} as {predicted}, expected {true_label}")
            return False

    def evaluate(self, classifier, test_df, target_column):
        correct_predictions = 0
        total_predictions = len(test_df)
        for index, row in test_df.iterrows():
            record = row.drop(target_column).to_dict()
            predicted = classifier.classify(record)
            if predicted == row[target_column]:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {correct_predictions}/{total_predictions} ({(correct_predictions / total_predictions) * 100:.2f}%)")
        return accuracy
