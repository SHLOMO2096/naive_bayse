def classify(model, record):
    priors = model["priors"]
    classes = list(priors.keys())

    probs = {cls: priors[cls] for cls in classes}

    for col in record:
        value = record[col]
        for cls in classes:
            try:
                probs[cls] *= model[col][value][f'P({cls}|X)']
            except KeyError:
                probs[cls] *= 1e-6
    return max(probs, key=probs.get)
