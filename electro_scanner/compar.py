import csv
from difflib import SequenceMatcher
from multiprocessing import Pool

def compare(string_pair):
    string1, string2 = string_pair
    similarity = SequenceMatcher(None, string1, string2).ratio()
    return similarity, string2

def compare_to_database(user_input):

    similar_elements = []

    with open('elements.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        elements = [row[0].rstrip(';') for row in reader if row and row[0].rstrip(';')]

    if elements:
        # Use multiprocessing Pool to parallelize similarity comparison
        with Pool() as pool:
            results = pool.map(compare, [(user_input, elem) for elem in elements])

        # Filter similar elements
        for similarity, element in results:
            if similarity >= 0.85:
                similar_elements.append(element)

    if similar_elements:
        most_similar = max(similar_elements, key=lambda x: sum(1 for i, j in zip(user_input, x) if i == j))
        print(f"Most similar element: {most_similar}")
        return most_similar
    else:
        print("No similar elements.")
        return None

