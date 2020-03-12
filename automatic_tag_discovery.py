import pickle
import time

from data.data import read_large_file, parse_business
from aspect_opinion_extraction.pairing import pairing
from index import make_tag
from similarity.semantic_sim.embedding_model import paragram_embedding
from similarity.semantic_sim.features import cosine_sim


class TagCentroid:
    def __init__(self, tag, embedding):
        self.embedding = embedding
        self.phrases = [tag]
        self.entities = {}


    def add_to_mean(self, tag, embedding):
        """
            Adds embedding to the mean embedding of this centroid
            and apend tag to the list of textual tags of the centroid
        """
        n = len(self.phrases)
        self.embedding = (1/n+1) * embedding + (n/n+1) * self.embedding
        self.phrases.append(tag)


    def text_name(self):
        "From the list of phrases, return the phrase that is closest to the embedding of this centroid tag"
        return max([(p, cosine_sim(self.embedding, paragram_embedding(p))) for p in self.phrases], key=lambda x: x[1])[0]





def discover(reviews_filename, result_filename, sim_threshold=0.5, layer=8, head=5, max_review_length=512):

    centroid_tags = []

    with open(reviews_filename, 'r') as f:
        for business in read_large_file(f):
            business_id, reviews = parse_business(business)
            start = time.time()

            current_tag_counts = [0] * len(centroid_tags)

            for r in reviews:
                current_tag_visited = [False] * len(centroid_tags)

                pairs = pairing(r[:max_review_length], layer, head)
                for p in pairs:
                    tag = make_tag(p)
                    tag_embedding = paragram_embedding(tag)
                    new_centroid = True
                    for i, c in enumerate(centroid_tags):
                        try:
                            sim = cosine_sim(tag_embedding, c.embedding)
                            if sim > sim_threshold:
                                if not current_tag_visited[i]:
                                    current_tag_visited[i] = True
                                    current_tag_counts[i] += 1

                                    c.add_to_mean(tag, tag_embedding)
                                    new_centroid = False
                        except:
                            print("\nERROR ==> business_id = {}  |  pair = {}  | review = {}\n".format(business_id, make_tag(p), r))

                    if new_centroid:
                        centroid_tags.append(TagCentroid(tag, tag_embedding))
                        current_tag_visited.append(False)
                        current_tag_counts.append(0)

            # Normalize the counts into percentages
            current_tag_counts = [c/len(reviews) for c in current_tag_counts]

            for i, c in enumerate(centroid_tags):
                if current_tag_counts[i] != 0:
                    c.entities[business_id] = current_tag_counts[i]

            with open(result_filename + "_list", 'wb') as f:
                pickle.dump(centroid_tags, f)
            print("{} Done in {} seconds".format(business_id, time.time() - start))

    new_tags = {c.text_name:c.entities for c in centroid_tags}
    with open(result_filename, 'wb') as f:
        pickle.dump(new_tags, f)





# with open("data/index/index.pkl", "rb") as f:
#     index = pickle.load(f)


if __name__ == "__main__":
    reviews_filename = "data/reviews/processed/reviews.txt"
    result_filename = "data/index/automatic.pkl"
    discover(reviews_filename, result_filename)