import numpy as np

import model
class NaiveBayesClassifier:
    def __init__(self):
        self.model = {}
        self.prior_1 = 0
        self.prior_2 = 0

    def fit(self, df, target_column):
        target_variable = df[target_column]
        feature_cols = df.drop(columns=[target_column],axis=1)
        unique_target_variable = target_variable.unique()
        print(type(unique_target_variable))
        prior = df[target_column].value_counts(normalize=True).to_dict()
        # self.prior_yes = prior[0]
        # self.prior_no = prior[1]

        for col in feature_cols:
            self.model[col] = {}
            unique_values = feature_cols[col].unique()
            for value in unique_values:
                yes_count = len(df[(df[col] == value) & (target_variable == unique_target_variable[0])])
                no_count = len(df[(df[col] == value) & (target_variable == unique_target_variable[0])])
                total_yes = (target_variable == unique_target_variable[0]).sum()
                total_no = (target_variable == unique_target_variable[1]).sum()

                self.model[col][value] = {
                    'P(YES|X)': yes_count / total_yes,
                    'P(NO|X)': no_count / total_no
                }

        print(self.model)

    def classify(self, record):
        prob_yes = self.prior_1
        prob_no = self.prior_2

        for col in record:
            value = record[col]
            if value in self.model[col]:
                prob_yes *= self.model[col][value]['P(YES|X)']
                prob_no *= self.model[col][value]['P(NO|X)']

        if prob_yes > prob_no:
            return 'yes'
        else:
            return 'no'


a = NaiveBayesClassifier()
a.fit(model.df,'p')