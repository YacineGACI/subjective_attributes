def read_large_file(file_object):
    """
        Uses a generator to read a large file lazily, line by line
    """
    while True:
        line = file_object.readline()
        if not line:
            break
        yield line



def get_categories(business_line):
    """
        Returns the categories of the business in @business_line
        @business_line is a json object
    """
    if not business_line["categories"]:
        return "None"
    return business_line["categories"].split(", ")



def get_city(business_line):
    """
        Returns the city of the business in @business_line
        @business_line is a json object
    """
    return business_line["city"]


def get_review(review_line):
    """
        Returns the text review from the review object
        @reviews_line is a json object
    """
    return review_line["text"]


def parse_review(review, placeholder=" "):
    """
        @review is a string
        Replaces \t \n and \r with @placeholder
    """
    return review.replace("\t", placeholder).replace("\n", placeholder).replace("\r", " ")


def parse_review_to_disk(business_id, reviews):
    """
        @reviews is a list of strings
        Returns a long string containing @business_id and tab-separated reviews, ready to write to disk
    """
    return business_id + "\t" + "\t".join([parse_review(r) for r in reviews]) + "\n"


def parse_business(business_line):
    """
        @business_line is a tab-separated string whose first element is the business_id and the rest are the list of reviews
    """
    parts = business_line.split('\t')
    return parts[0], parts[1:]