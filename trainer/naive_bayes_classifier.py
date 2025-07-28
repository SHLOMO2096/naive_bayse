class NaiveBayesClassifier:
    def __init__(self):
        self.model = {}
        self.priors = {}
        self.classes = []


    def fit(self, df, target_column):
        target_variable = df[target_column]
        feature_cols = df.drop(columns=[target_column],axis=1)
        self.classes = target_variable.unique()
        self.priors = df[target_column].value_counts(normalize=True).to_dict()
        self.model["priors"] = self.priors

        for col in feature_cols:
            self.model[col] = {}
            unique_values = feature_cols[col].unique()
            for value in unique_values:
                self.model[col][value] = {}
                yes_count = len(df[(df[col] == value) & (target_variable == self.classes[0])]) +1
                no_count = len(df[(df[col] == value) & (target_variable == self.classes[1])]) +1
                self.model[col][value] = {self.classes[0]: yes_count, self.classes[1]: no_count}

                total_yes = (target_variable == self.classes[0]).sum() + len(unique_values)
                total_no = (target_variable == self.classes[1]).sum() + len(unique_values)

                self.model[col][value][f'P({self.classes[0]}|X)'] = yes_count / total_yes
                self.model[col][value][f'P({self.classes[1]}|X)'] = no_count / total_no

    def classify(self, record):
        priors = self.model["priors"]
        classes = list(priors.keys())

        probs = {cls: priors[cls] for cls in classes}

        for col in record:
            value = record[col]
            for cls in classes:
                try:
                    probs[cls] *= self.model[col][value][f'P({cls}|X)']
                except KeyError:
                    probs[cls] *= 1e-6
        return max(probs, key=probs.get)








