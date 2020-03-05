import json
import pickle

from data import *


if __name__ == "__main__":

    businesses = {}
    categories = {}
    locations = {}
    with open("data/reviews/raw/business.json", 'r') as f:
        for line in read_large_file(f):
            business = json.loads(line)

            # Add business to te business dictionary
            businesses[business["business_id"]] = business

            # Add this business to the corresponding categories
            for c in get_categories(business):
                if c in categories.keys():
                    categories[c].append(business["business_id"])
                else:
                    categories[c] = [business["business_id"]]

            # Add this business to the corresponding city/location
            city = get_city(business)
            if city in locations.keys():
                locations[city].append(business["business_id"])
            else:
                locations[city] = [business["business_id"]]
    
    # Serialize businesses, categories and locations dicts into disk
    with open("data/reviews/processed/businesses.pkl", "wb") as f:
        pickle.dump(businesses, f)

    with open("data/reviews/processed/categories.pkl", "wb") as f:
        pickle.dump(categories, f)

    with open("data/reviews/processed/locations.pkl", "wb") as f:
        pickle.dump(locations, f)


    # A dict whose keys are @business_ids, and values are lists of corresponding reviews
    reviews = {}
    with open("data/reviews/raw/review.json", "r") as f:
        for line in read_large_file(f):
            review = json.loads(line)
            text = get_review(review)
            if review['business_id'] in reviews.keys():
                reviews[review['business_id']].append(text)
            else:
                reviews[review['business_id']] = [text]
    
    # Serialize the reviews into a txt file in order to be able to use generators while indexing the reviews
    with open("data/reviews/processed/reviews.txt", "w", encoding='utf-8') as f:
        for r in reviews.keys():
            f.write(parse_review_to_disk(r, reviews[r]))

