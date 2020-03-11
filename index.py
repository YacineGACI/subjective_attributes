import pickle
import os, time

from data.data import read_large_file, parse_business
from aspect_opinion_extraction.pairing import pairing
from similarity.compute_similarity import similarity


def make_tag(pair):
    aspect, opinion = pair
    return aspect + " " + opinion




def business_in_category(business_id, cats):
    """
        @cats: list of categories to consider
        If @business_id has at least one of the categories specified in @cats then return True
        Else return False
    """
    for c in cats:
        if business_id in categories[c]:
            return True
    return False



def inverted_index(tags, review_filename, index_filename, theta, cats=None, layer=8, head=5, max_review_length=512):
    """
        @tags: list of initial tags to include in the inverted index
        @review_filename: file of business reviews per line
        @index_filename: file where to save the final inverted index
        @theta: threshold of similarity between tags
        @cats: If None, index all businesses. Otherwise, only keep those whose categories are in cats

        Builds an inverted index for the reviews where the keys are available tags
        The postings of the inverted index are themselves dicts whose keys are business_ids and values are scores

        The index is of the following structure
        {
            tag_1:{
                business_id: score,
                business_id: score,
                business_id: score,
                ...
            },
            tag_2:{
                business_id: score,
                business_id: score,
                ...
            },
            ...
        }
    """
    tag_dict = None

    # If @index_file already exists, load it
    if os.path.isfile(index_filename):
        with open(index_filename, 'rb') as f:
            tag_dict = pickle.load(f)
    # Otherwise, create a new empty dict
    else:
        tag_dict = {t:{} for t in tags} 

    with open(review_filename, 'r') as f:
        for business in read_large_file(f):
            start = time.time()
            business_id, reviews = parse_business(business)

            if cats is not None:
                if not business_in_category(business_id, cats):
                    continue

            current_tag_counts = [0] * len(tags) # Tag counts for this business. Each cell is the count of the correspondign tag present in the current business reviews

            for r in reviews:
                current_tag_visited = [False] * len(tags) # Useful for not to count the same tag in the same review twice
                pairs = pairing(r[:max_review_length], layer, head)
                for p in pairs:
                    for i, t in enumerate(tags):
                        try:
                            if similarity(t, make_tag(p)) > theta:
                                if not current_tag_visited[i]:
                                    current_tag_visited[i] = True
                                    current_tag_counts[i] += 1
                        except:
                            print("\nERROR ==> business_id = {}  |  pair = {}  | tag = {} | review = {}\n".format(business_id, make_tag(p), t, r))

            # Normalize the counts into percentages
            current_tag_counts = [c/len(reviews) for c in current_tag_counts]

            for i, t in enumerate(tags):
                if current_tag_counts[i] != 0:
                    tag_dict[t][business_id] = current_tag_counts[i]
            

            with open(index_filename, 'wb') as f:
                pickle.dump(tag_dict, f)
            print("{} Done in {} seconds".format(business_id, time.time() - start))

    



if __name__ == "__main__":

    with open("data/reviews/processed/businesses.pkl", 'rb') as f:
        businesses = pickle.load(f)

    with open("data/reviews/processed/categories.pkl", 'rb') as f:
        categories = pickle.load(f)

    tags = ["food great", "staff helpful", "plates clean", "price fair", "parking free"]
    inverted_index(tags, "data/reviews/processed/reviews.txt", "data/index/index.pkl", 0.4)