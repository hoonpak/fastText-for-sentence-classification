def essential_args(data_type):
    if data_type == "ag_news_csv":
        return False, 0.25
    elif data_type == "sogou_news_csv":
        return True, 0.5
    elif data_type == "dbpedia_csv" :
        return False, 0.5
    elif data_type == "yelp_review_polarity_csv" :
        return False, 0.1
    elif data_type == "yelp_review_full_csv" :
        return False, 0.1
    elif data_type == "yahoo_answers_csv" :
        return False, 0.1
    elif data_type == "amazon_review_full_csv" :
        return False, 0.05
    elif data_type == "amazon_review_polarity_csv" :
        return False, 0.05