import json
from classifier import strict_classifier, unrestricted_classifier, conerhole_free_classifier

def load_classification_table(file_path):
    """
    Load the classification table from a JSON file.
    """
    with open(file_path, 'r') as f:
        classification_table = json.load(f)  # Load the JSON file
    return classification_table

def poly_classifier(classification_file_path, polygons_file_path, output_file_path, strategy="strict"):
    """
    Classify polygons based on a classification table and output the results.

    Parameters:
    - classification_file_path: Path to the classification table file.
    - polygons_file_path: Path to the polygons file to be classified.
    - output_file_path: Path to the file where classification results will be written.
    - strategy: Classification strategy, either "strict" or "unrestricted".
    """
    # Load classification table
    classification_table = load_classification_table(classification_file_path)

    # Load polygons to be classified
    with open(polygons_file_path, 'r') as f:
        polygons = json.load(f)

    # Initialize results list
    results = []

    # Iterate over each polygon and classify
    for keyname, polygon_data in polygons.items():
        cornerhole_num = 0
        free_edges_sequence = polygon_data["free_edges_sequence"]
        reversed_free_edges_sequence = free_edges_sequence[::-1]
        edges_sequence = polygon_data["non_free_edges_sequence"]
        reversed_edges_sequence = [
            [edge[0], list(reversed(edge[1]))] for edge in reversed(edges_sequence)
        ]

        # Select the classifier based on strategy
        if strategy == "strict":
            matched_type = strict_classifier(
                classification_table, cornerhole_num, free_edges_sequence,
                reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence
            )
        elif strategy == "unrestricted":
            matched_type = unrestricted_classifier(
                classification_table, cornerhole_num, free_edges_sequence,
                reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence
            )
        elif strategy == "conerhole_free":
            matched_type = conerhole_free_classifier(
                classification_table, cornerhole_num, free_edges_sequence,
                reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence
            )
        else:
            raise ValueError(f"Unknown classification strategy: {strategy}")

        # Append the result in A:B format
        results.append(f"{keyname}:{matched_type}")

    # Write results to output file
    with open(output_file_path, 'w') as f:
        f.write("\n".join(results))

    print(f"Classification results written to {output_file_path}")

if __name__ == '__main__':
    classification_file_path = "./type.json"
    polygons_file_path = "./output/bracket.json"
    output_file_path = "./output/classifier_res.txt"
    poly_classifier(classification_file_path, polygons_file_path, output_file_path, strategy="conerhole_free")