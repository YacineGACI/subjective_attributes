import pickle

from similarity.compute_similarity import similarity

def filter_entities(user_tags, sim_threshold=0.3, to_index_later="data/index/to_index.txt"):
    """
        @user_tags: list of tags the user is interested in
        @index: pre-built index of tags
        @sim_threshold: a similarity threshold used when comparing user tag with index tags
        @to_index_later: filepath to a txt file that stores all tags that are not present in the current index but the previous users asked about

        Returns the list of entities that have the tags specified by the user along with their ranking scores
    """
    res_dict = {t:{} for t in user_tags} # Each user tag has a dict of all entities having the tag with a value of truth
    to_be_indexed = [] # List of user tags not present in the index

    for t in user_tags:
        if t in index.keys():
            # This tag already exists in the index
            res_dict[t] = index[t]
        else:
            # This tag does not exist in the index
            # We take the union set postings of all similar tags in the index
            # For the value of truth of this tags, they will be a weighted sum of similar tags degrees of truth where the weights are the similarity scores themselves
            to_be_indexed.append(t)
            sim_sum = 0
            for index_tag in index.keys():
                sim = similarity(t, index_tag)
                if  sim > sim_threshold:
                    sim_sum += sim
                    for entity in index[index_tag].keys():
                        if entity in res_dict[t]:
                            # If the entity is already present. This usually happens when other tags have this entity and we already added it to this dict
                            res_dict[t][entity] += sim * index[index_tag][entity]
                        else:
                            # If the entity is added for the first time
                            res_dict[t][entity] = sim * index[index_tag][entity]
            
            # Divide by the total similarity to make it a weighted average of degrees of truths
            if sim_sum > 0:
                for k in res_dict[t].keys():
                    res_dict[t][k] /= sim_sum

    
    with open(to_index_later, 'a') as f:
        f.write("\n".join(to_be_indexed) + "\n")
    
    return res_dict






def combine_tags(tag_dict):
    """
        @tag_dict: a dictionary whose keys are the tags the user asked about and the values are entities with their degrees of truth

        Comptes the intersection of entities of all tags and combines the degrees of truth
        Returns a dict whose keys are business_ids and values are final degrees of truth
    """

    tags = list(tag_dict.keys())
    result = {}
    for entity in tag_dict[tags[0]]:
        # For each entity in the list of entities of the first tag

        score_for_this_entity = tag_dict[tags[0]][entity] #Intialize the final score
        num_visited_tags = 1 # To check the number of tags that have this entity. Must be equal of the number of all tags in order to consdier this entity in the final result
        
        for t in tags[1:]:
            this_tag_has_this_entity = False
            for e in tag_dict[t].keys():
                if entity == e:
                    this_tag_has_this_entity = True
                    score_for_this_entity += tag_dict[t][e]
                    num_visited_tags += 1
                    break
            
            if this_tag_has_this_entity == False:
                break

        if num_visited_tags == len(tags):
            result[entity] = score_for_this_entity / num_visited_tags

    return result
        


def filter_and_rank(user_tags, selected_entities):
    """
        @user_tags: the subjective tags the user is interested in
        @selected_entities: list of entities that have already been filtered by objective attributes

        Selects the entities in the index that have all subjective tags from the user and are present in @selected_entities
        And ranks them according to their final degrees of truth
    """
    entities_per_tag = filter_entities(user_tags)
    subj_entities = combine_tags(entities_per_tag)
    final_results = [(k, subj_entities[k]) for k in subj_entities.keys() if k in selected_entities]
    return sorted(final_results, reverse=True, key=lambda x: x[1])




# Read the index dict
with open("data/index/index.pkl", 'rb') as f:
    index = pickle.load(f)




if __name__ == "__main__":
    print("Filtering began")
    user_tags = ["chicken wings delicious", "staff friendly"]
    entities_per_tag = filter_entities(user_tags)
    print(entities_per_tag)
    subj_entities = combine_tags(entities_per_tag)
    print(subj_entities)
    