# Function to classify a record using a Naive Bayes model
# model - the trained model (dict)
# record - the record to classify (dict)
def classify(model, record):
    # Get prior probabilities for each class
    priors = model["priors"]
    classes = list(priors.keys())

    # Initialize probability dictionary for each class
    probs = {cls: priors[cls] for cls in classes}

    # Iterate over each column in the record
    for col in record:
        value = record[col]
        for cls in classes:
            try:
                # Update the probability for the class based on the column value
                probs[cls] *= model[col][value][f'P({cls}|X)']
            except KeyError:
                # If the value is not in the model, use a very small probability
                probs[cls] *= 1e-6
    # Return the class with the highest probability
    return max(probs, key=probs.get)
