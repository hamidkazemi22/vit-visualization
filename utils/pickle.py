import pickle


def load_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as file:
        return pickle.load(file)
