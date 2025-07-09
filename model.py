import pandas as pd
from numpy.ma.extras import unique

df = pd.read_csv('agaricus-lepiota.csv')

target_variable = df.iloc[:,-1]
feature_cols = df.iloc[:,:-1]
# print(df.head())

buying_a_computer = (target_variable == 'yes').sum() / len(df.iloc[:,-1])
dont_buy_a_computer = (target_variable == 'no').sum() / len(df.iloc[:,-1])

model = {}
for col in feature_cols:
    model[col] = {}
    unique_values = unique(df[col])
    for value in unique_values:
        model[col][value] = {}
        yes_count = len(df[(df[col] == value) & (target_variable == 'yes')])
        no_count = len(df[(df[col] == value) & (target_variable == 'no')])
        model[col][value] = {'YES': yes_count, 'NO': no_count}

        total_yes = (target_variable == 'yes').sum()
        total_no = (target_variable == 'no').sum()
        model[col][value]['P(YES|X)'] = yes_count / total_yes
        model[col][value]['P(NO|X)'] = no_count / total_no

# print(model)

# Classify a new record using the Naive Bayes model
def classify(record):
    prob_yes = buying_a_computer
    prob_no = dont_buy_a_computer

    for col in record:
        value = record[col]
        if value in model[col]:
            prob_yes *= model[col][value]['P(YES|X)']
            prob_no *= model[col][value]['P(NO|X)']

    if prob_yes > prob_no:
        return 'yes'
    else:
        return 'no'


# # Example record to classify
# record = {
#     'age': '<=30',
#     'income': 'medium',
#     'student': 'yes',
#     'credit_rating': 'fair'
# }
#
# # Classify the example record
# classification = classify(record)
# print(f"The classification for the record {record} is: {classification}")


corect = 0
total = len(feature_cols)
for i, row in feature_cols.iterrows():
    record = row.to_dict()
    predicted = classify(record)
    actual = target_variable.iloc[i]
    if predicted == actual:
        corect += 1
print(corect)
print(f"Correct: {corect}/{total} ({(corect/total)*100:.2f}%)")



