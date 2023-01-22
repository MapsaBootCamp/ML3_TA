import pandas as pd


def load_data(file_address='./data/processed.cleveland.data'):
    binary = pd.CategoricalDtype(categories=[0, 1])
    df = pd.read_csv(
        file_address,
        header=None,
        na_values='?',
        names='age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target'.split(', '),
        dtype={
            'sex':binary,
            'cp':pd.CategoricalDtype(categories=range(1,5)),
            'fbs':binary,
            'restecg':pd.CategoricalDtype(categories=range(3)),
            'exang':binary,
            'slope':pd.CategoricalDtype(categories=range(1,4)),
            'ca':pd.CategoricalDtype(categories=range(4)),
            'thal':pd.CategoricalDtype(categories=[3,6,7]),
            'target':pd.CategoricalDtype(categories=range(2)),
        }
    )

    return df.fillna({'ca':0, 'thal':3, 'target':1})