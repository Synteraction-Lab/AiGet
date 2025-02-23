import time
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def find_most_relevant_history(context_description, history_list, top_n=10):
    """
    Find the most relevant history items based on the provided context.

    :param context_json: JSON object containing contextual information.
    :param history_list: List of history items.
    :param top_n: Number of top relevant items to return.
    :return: List of the most relevant history items.
    """
    # Extract and encode the context description
    if history_list is None or len(history_list) < 10:
        return history_list

    start_time = time.time()
    context_description = str(context_description)
    context_embedding = model.encode([context_description], convert_to_tensor=True)

    # Encode all history items
    history_embeddings = model.encode(history_list, convert_to_tensor=True)

    # Compute cosine similarities between context and history items
    similarities = util.cos_sim(context_embedding, history_embeddings)[0]

    # Get the indices of the top N most similar items
    top_indices = similarities.topk(k=top_n).indices.cpu().numpy()

    # Retrieve the corresponding history items
    most_relevant_items = [history_list[i] for i in top_indices]
    print(f"Filter History Time: {time.time() - start_time}")

    return most_relevant_items

def filter_new_items(new_item_list, history_list, threshold=0.7):
    """
    Compare new_item_list with history_list and return items that are
    sufficiently different to be added to history. If too similar,
    return the most similar item from history.

    :param new_item_list: List of new items to compare.
    :param history_list: List of existing history items.
    :param threshold: Similarity threshold to consider items as too similar.
    :return: List of new items or the most similar item from history.
    """
    # Encode history items
    if history_list is None or len(history_list) == 0:
        return new_item_list

    history_embeddings = model.encode(history_list, convert_to_tensor=True)

    filtered_items = []

    for new_item in new_item_list:
        # Encode the new item
        new_embedding = model.encode([new_item], convert_to_tensor=True)

        # Calculate cosine similarities between the new item and all history items
        similarities = util.cos_sim(new_embedding, history_embeddings)

        # Find the maximum similarity and the corresponding history item
        max_similarity = similarities.max().item()
        max_similarity_index = similarities.argmax().item()
        most_similar_item = history_list[max_similarity_index]

        # Check if the similarity is below the threshold
        if max_similarity < threshold:
            filtered_items.append(new_item)
            # print(f"[OK] {new_item} is similar to: \n{most_similar_item}\n\n")
        else:
            print(f"[BAD] {new_item} is similar to: \n{most_similar_item}\n\n")
            # filtered_items.append(most_similar_item)

    return filtered_items

def test_filter_new_items():
    """Test the filter_new_items function by loading history and resources, comparing them, and updating the history."""
    # Path to the history JSON file
    from src.Utilities.process_recording_data import generate_ai_suggestions_file_from_pid
    import os
    PID = "p18"
    generate_ai_suggestions_file_from_pid(PID)
    root_dir = '/Users/Vincent/Documents/GitHub/PandaExp/data/recordings'
    history_file_path = os.path.join(root_dir, f'{PID}', 'ai_suggestions.json')

    # Load history from JSON file
    with open(history_file_path, 'r') as file:
        resources_list = json.load(file)

    # Load resources from a different source if needed, here we assume it's loaded the same way as history
    history_list = []

    while resources_list:
        # Pick the first item from the resources_list
        start_time = time.time()
        new_item = resources_list.pop(0)

        # Compare it with history_list and filter
        filtered_items = filter_new_items([new_item], history_list, threshold=0.75)

        # Add the returned item (either the new item or the most similar history item) to the history_list
        if filtered_items:
            history_list.append(filtered_items[0])
        print(f"Len History: {len(history_list)}, Processed time: {time.time() - start_time}")

    # Save the updated history back to the JSON file
    with open(history_file_path, 'w') as file:
        json.dump(history_list, file, indent=4)

if __name__ == '__main__':
    test_filter_new_items()
