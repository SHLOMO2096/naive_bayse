
class Evaluator:
    # Evaluate the classifier on a test DataFrame
    def evaluate(self, classifier, test_df, target_column):
        correct_predictions = 0
        total_predictions = len(test_df)
        for index, row in test_df.iterrows():
            # Prepare the record without the target column
            record = row.drop(target_column).to_dict()
            predicted = classifier.classify(record)
            if predicted == row[target_column]:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        # Return accuracy as a formatted string
        return f"Accuracy: {correct_predictions}/{total_predictions} ({(correct_predictions / total_predictions) * 100:.2f}%)"
