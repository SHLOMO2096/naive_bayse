class NaiveBayesClassifier:
    def __init__(self):
        # Model stores the learned probabilities
        self.model = {}
        self.priors = {}
        self.classes = []

    def fit(self, df, target_column):
        # Fit the Naive Bayes model to the data
        target_variable = df[target_column]
        feature_cols = df.drop(columns=[target_column],axis=1)
        self.classes = target_variable.unique()
        # Calculate prior probabilities for each class
        self.priors = df[target_column].value_counts(normalize=True).to_dict()
        self.model["priors"] = self.priors

        # Calculate conditional probabilities for each feature value
        for col in feature_cols:
            self.model[col] = {}
            unique_values = feature_cols[col].unique()
            for value in unique_values:
                self.model[col][value] = {}
                # Laplace smoothing (+1)
                yes_count = len(df[(df[col] == value) & (target_variable == self.classes[0])]) +1
                no_count = len(df[(df[col] == value) & (target_variable == self.classes[1])]) +1
                self.model[col][value] = {self.classes[0]: yes_count, self.classes[1]: no_count}

                total_yes = (target_variable == self.classes[0]).sum() + len(unique_values)
                total_no = (target_variable == self.classes[1]).sum() + len(unique_values)

                # Store conditional probabilities for each class
                self.model[col][value][f'P({self.classes[0]}|X)'] = yes_count / total_yes
                self.model[col][value][f'P({self.classes[1]}|X)'] = no_count / total_no

    def classify(self, record):
        # Classify a new record using the trained model
        priors = self.model["priors"]
        classes = list(priors.keys())

        probs = {cls: priors[cls] for cls in classes}

        for col in record:
            value = record[col]
            for cls in classes:
                try:
                    # Multiply by conditional probability for each feature
                    probs[cls] *= self.model[col][value][f'P({cls}|X)']
                except KeyError:
                    # Use a very small probability if value not seen in training
                    probs[cls] *= 1e-6
        # Return the class with the highest probability
        return max(probs, key=probs.get)








