import requests
from aspect_opinion_extraction.pairing import pairing


def make_tag(pair):
    aspect, opinion = pair
    return aspect + " " + opinion


def get_intent_slots(query, target_intent="FindRestaurant"):

    res = requests.get("https://api.wit.ai/message?v=20190312&q=" + query, headers=headers)
    entities = res.json()['entities']
    intent = entities['intent'][0]["value"]

    if intent == target_intent:
        return {k:v[0]["value"] for k,v in entities.items()}
    else:
        return None



def extract_subjective_attributes(query, layer=8, head=5):
    pairs = pairing(query, layer, head)
    return [make_tag(p) for p in pairs]



def parse_user_input(query, target_intent="FindRestaurant"):
    result = get_intent_slots(query, target_intent)
    result["subjective_attributes"] = extract_subjective_attributes(query)
    return result


headers = {'Authorization': 'Bearer DVQSW5GF6K5AXEABMVM3W6JVAIRGPNZ3'}


if __name__ == "__main__":

    query = "Find me french restaurants in Sydney that serve great seafood and has a friendly staff and clean dishware"
    # query = "French restaurant"

    print(parse_user_input(query))