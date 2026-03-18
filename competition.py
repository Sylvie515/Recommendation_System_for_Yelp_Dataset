"""
Method Description:
I build a hybrid recommendation system that combines item-based CF and a model-based XGBoost regressor.

(1) Item-based CF:
    - Build [(user_id, business_id, stars), ......] records from yelp_train.csv.
    - Filter out users with only 1 rating or more than 500 ratings, and keep only businesses with at least 100 ratings when constructing LSH signatures.
    - MinHash + LSH with 20 bands and 5 rows per band generates candidate business pairs.
    - Compute Pearson similarity for candidate pairs, and scale it by min(count, 60)/60 to downweight pairs with very few co-rated users.
    - For each business, keep at most top-35 similar neighbors.
    - For prediction, only neighbors that the target user has rated are used. The CF score is a similarity-weighted average, then lightly smoothed toward the global average and clipped to [1, 5].
(2) XGBoost model:
    - Extract additional features from user.json and business.json. Non-finite feature values (NaN/inf) are converted to 0.0.
    - Hyperparameters are tuned by offline GridSearch CV on a subsample of training data:
      • 2-stage: A coarse search over a wider grid, followed by a finer search around the best region.
      • Both use 5-fold CV and RMSE metric, searching over learning_rate, max_depth, n_estimators, min_child_weight, reg_alpha, etc.
    - Several CV parameters achieved similar RMSE around 0.98. The final parameters are chosen based on the performance of the full CF + XGBoost hybrid pipeline on validation data, rather than purely on the standalone XGBoost CV scores.
    - XGBoost outputs are smoothed toward the global average and clipped to [1, 5].
(3) Hybrid scheme:
    - For each (user, business) pair in the test set:
      • If the target business has at least 25 CF neighbors and a CF prediction exists:
        > Mix a neighbor-count–weighted CF score with a simple baseline.
        > Then combine CF score with XGBoost score using an alpha that grows with log(neighbor_cnt) (capped at 0.2).
      • Otherwise, use the XGBoost prediction only.
    - Apply a global smoothing and clip the result into [1, 5] as final prediction.

Error Distribution:
>=0 and <1: 102076
>=1 and <2: 33047
>=2 and <3: 6155
>=3 and <4: 766
>=4       : 0

RMSE: 0.9781699160686848

Execution Time: 490.0219190120697 seconds
"""



from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
import sys
import math
from itertools import combinations
import random
import json
import xgboost as xgb
import numpy as np



def get_PearsonSimilarity(key_value, business_avg, global_avg):
    # pearson_rdd = ((business_i, bussiness_j), (count, sum_rating_i, sum_rating_j, sum_rating_i_sqr, sum__rating_j_sqr, sum__rating_i__rating_j))
    (business_i, business_j), (count, sum_ri, sum_rj, sum_ri_sqr, sum_rj_sqr, sum_ri_rj) = key_value

    # if fewer than 2 users co-rated the business pairs, return 0.0 (similarity is unreliable)
    if count < 2:
        return ((business_i, business_j), 0.0)

    # business i & j avg. rating; if no business avg. rating, use global rating
    avg_i = business_avg.get(business_i, global_avg)
    avg_j = business_avg.get(business_j, global_avg)

    # numerator = Σ ((r_i - avg_i) * (r_j - avg_j))
    numerator = sum_ri_rj - avg_j * sum_ri - avg_i * sum_rj + count * avg_i * avg_j
    # denominators = (Σ (r_i - avg_i)^2)^0.5 * (Σ (r_j - avg_j)^2)^0.5
    r_i_sqr = sum_ri_sqr + count * avg_i**2 - 2 * avg_i * sum_ri
    r_j_sqr = sum_rj_sqr + count * avg_j**2 - 2 * avg_j * sum_rj

    if r_i_sqr <= 0 or r_j_sqr <= 0:
        return ((business_i, business_j), 0.0)
    denominator = math.sqrt(r_i_sqr) * math.sqrt(r_j_sqr)
    if denominator == 0:
        PearsonSimilarity = 0.0
    else:
        # restrict similarity in [-1, 1]
        PearsonSimilarity = max(-1.0, min(1.0, numerator / denominator))

    ## cold-start problem: reduce the weight of businesses that have fewer than 60 users corated (n_corating = count)
    scaling = min(count, 60) / 60.0
    PearsonSimilarity = PearsonSimilarity * scaling

    return ((business_i, business_j), PearsonSimilarity)



def predict_rating(join_rdd, business_avg, global_avg):
    # business_i
    business_i = join_rdd[0]
    # (user_id, [(business_j, stars_j), (business_k, stars_k), ......])
    user_data = join_rdd[1][0]
    # user_id
    user_id = user_data[0]
    # [(business_j, stars_j), (business_k, stars_k), ......] => {business_j: stars_j, business_k: stars_k, ......}
    user_rating_dict = dict(user_data[1])

    # avg. rating for business_id
    # cold-start problem: if no business_i avg. rating, use global rating as predicted_rating
    business_i_avg = business_avg.get(business_i, global_avg)

    # neighbors of business_i
    neighbors = join_rdd[1][1]    # None or [(business_j, stars_j), (business_k, stars_k), ......]
    # cold-start problem: if business_i has no neighbors
    if not neighbors:
       return (user_id, business_i, business_i_avg)
    business_neighbors = dict(neighbors)
        
    # find the businesses 1. user_id has rated & 2. is business_id's neighbor
    common_business = set(user_rating_dict.keys()) & set(business_neighbors.keys())

    # cold-start problem: if no common_business, use business_avg as predicted_rating
    if len(common_business) == 0:
        return (user_id, business_i, business_i_avg)

    ## making predictions

    # numerator = Σ rating_j * similarity_ij, denominators = |Σ similarity_ij|
    numerator = 0.0
    denominator = 0.0
    for business_j in common_business:
        # similarity_ij
        similarity_ij = business_neighbors[business_j]
        # use only non-zero correlations
        if similarity_ij != 0:
            # rating_j
            user_rating_j = user_rating_dict[business_j]
            # predict
            numerator += user_rating_j * similarity_ij
            denominator += abs(similarity_ij)

    if denominator == 0:
        return (user_id, business_i, business_i_avg)   
    
    rating_predict = numerator / denominator
    if not math.isfinite(rating_predict):    # NaN / inf
        rating_predict = business_i_avg
    smooth_factor = min(0.05 + len(common_business) / 400.0, 0.20)
    rating_predict = (1 - smooth_factor) * rating_predict + smooth_factor * global_avg
    # clipping: reduce RMSE
    rating_predict = max(1.0, min(5.0, rating_predict))

    return (user_id, business_i, rating_predict)



if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    # file path
    train_file_path = folder_path + "/data/yelp_train.csv"
    user_json_path = folder_path + "/data/user.json"
    business_json_path = folder_path + "/data/business.json"
    test_file_path = folder_path + "/data/" + test_file_name
    output_file_path = folder_path + "/output/" + output_file_name

    sc = SparkContext("local[*]", "task2_case3")
    sc.setLogLevel("ERROR")

    # read train_file (yelp_train.csv) as RDD
    yelp_rdd = sc.textFile(train_file_path)
    # no header
    header = yelp_rdd.first()
    yelp_rdd = yelp_rdd.filter(lambda r: r != header)
    # [(user_id, business_id, stars), (user_id, business_id, stars), ......]
    yelp_rdd = yelp_rdd.map(lambda r: r.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    yelp_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    # read test_file (yelp_val.csv) as RDD
    test_rdd = sc.textFile(test_file_path)
    # no header
    header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda r: r != header)
    # [(user_id, business_id), (user_id, business_id), ......]
    test_rdd = test_rdd.map(lambda r: r.split(",")).map(lambda x: (x[0], x[1]))
    test_rdd.persist(StorageLevel.MEMORY_AND_DISK)



    ## cold-start fallback & broadcast features

    # business_avg
    # if user is new or business has no neighbor: use business avg. rating (or global avg. rating if no business avg. rating)
        # (business_id, (stars, 1.0)) => (business_id, (total_stars, total_count))
    business_avg = yelp_rdd.map(lambda x: (x[1], (x[2], 1.0))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        # (business_id, (total_stars, total_count)) => (business_id, business_avg_stars)    
    business_avg = business_avg.mapValues(lambda x: x[0] / x[1])
    business_avg = business_avg.collectAsMap()
    business_avg_broadcast = sc.broadcast(business_avg)

    # user_avg
        # (user_id, (stars, 1.0)) => (user_id, (total_stars, total_count))
    user_avg = yelp_rdd.map(lambda x: (x[0], (x[2], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        # (user_id, (total_stars, total_count)) => (user_id, user_avg_stars)
    user_avg = user_avg.mapValues(lambda x: x[0] / x[1])
    user_avg = user_avg.collectAsMap()

    # global_avg
        # business_i is new; no business_avg: use global avg. rating
    global_avg = yelp_rdd.map(lambda x: x[2]).mean()
    global_avg_broadcast = sc.broadcast(global_avg)



    ## 1. item-based CF with Pearson similarity

    # user rating: (user_id, [list of (business_id, stars)])
        # [(user_1, [(business_1, stars_1), (business_2, stars_2), ...]), (user_2, [(business_1, stars_1), (business_2, stars_2), ...]), ......]
    user_rating_rdd = yelp_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list).persist(StorageLevel.MEMORY_AND_DISK)
        # drop users who rated only 1 or more than 500 businesses: (user_id, [(business_1, stars_1), (business_2, stars_2), ...])
    user_rating_rdd_1 = user_rating_rdd.filter(lambda x: 1 < len(x[1]) < 500)

    business_rating_count = yelp_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)
    valid_businesses = set(business_rating_count.filter(lambda x: x[1] >= 100).keys().collect())
    yelp_rdd_lsh = yelp_rdd.filter(lambda x: x[1] in valid_businesses)
    
    ## LSH
    
    # (k, {unique_v}): (business_id, {unique_user_id})
    business_user = yelp_rdd_lsh.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)
    # unique users (for hashing)
    user = business_user.flatMap(lambda x: list(x[1])).distinct().zipWithIndex()
    user = user.collectAsMap()
    user_broadcast = sc.broadcast(user)
    # user_id str => user_id int index: (user_id_str, user_id_int)
    business_user = business_user.mapValues(lambda user_id_set: {user_broadcast.value[user_id] for user_id in user_id_set if user_id in user_broadcast.value})
    user_broadcast.unpersist()

    # hash function: h(x) = (a * x + b) % M
    n_hashes = 100
    M = max(len(user) * 2 + 7, 1009)
    # n_hashes different (a, b) tuples => hash fuctions: a ∈ [1, M - 1], b ∈ [0, M - 1], a = b is acceptable
    used_hash_params = set()
    # hash_params = [(a1, b1), (a2, b2), ......]
    hash_params = []
    while len(hash_params) < n_hashes:
        a, b = random.randint(1, M - 1), random.randint(0, M - 1)
        if (a, b) not in used_hash_params:
            hash_params.append((a, b))
            used_hash_params.add((a, b))

    # signature: ((business_id, hash_index), min_hash_value)
    minhash_signature = business_user.flatMap(lambda x: [((x[0], i), min((a * user_id + b) % M for user_id in x[1])) for i, (a, b) in enumerate(hash_params)])

    # signature matrix
    # ((business_id, hash_index), min_hash_value) => (business_id, (hash_index, min_hash_value)) => (business_id, [(hash_index, min_hash_value), (hash_index, min_hash_value), ......])
    minhash_signature = minhash_signature.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey()
    # (business_id, [min_hash_value_0, min_hash_value_1, ......])      
    minhash_signature = minhash_signature.mapValues(lambda x: [hash_value for hash_index, hash_value in sorted(list(x))])

    b = 20
    r = 5
    # bucket_rdd = ((band_index, band_signature_tuple), business_id)
    bucket_rdd = minhash_signature.flatMap(lambda x: [((i, tuple(x[1][i * r : (i+1) * r])), x[0]) for i in range(b)])

    # candidate pairs
    # ((band_index, tuple(band_signature)), businesses hashed into the same bucket)
    candidate_rdd = bucket_rdd.groupByKey()
    # ((band_index, tuple(band_signature)), list[business_id])
    candidate_rdd = candidate_rdd.mapValues(list).filter(lambda x: len(x[1]) > 1)
    # combination
    candidate_pairs_lsh = candidate_rdd.flatMapValues(lambda v: list(combinations(sorted(v), 2))).map(lambda x: x[1]).distinct()
    # (business_i, business_j) as key: ((business_i, business_j), 1)
    lsh_candidates = candidate_pairs_lsh.map(lambda x: (x, 1))
    yelp_rdd.unpersist() 

    # co-rated business pairs by user
       # [(user_id, (business_1, stars_1)), (user_id, (business_2, stars_2)), ......] => [(business_1, stars_1), (business_2, stars_2), ......, (business_1, stars_1), (business_2, stars_2), ......]
    corated_business = user_rating_rdd_1.flatMapValues(lambda v: list(combinations(sorted(v), 2))).map(lambda x: x[1])
        # ((business_1, stars_1), (business_2, stars_2)) => ((business_1, business_2), (stars_1, stars_2))
    corated_business = corated_business.map(lambda x: ((x[0][0], x[1][0]), (x[0][1], x[1][1])))
    # keep only candidates found by LSH
    lsh_candidates = lsh_candidates.partitionBy(16)
    filtered_corated_business = corated_business.join(lsh_candidates).map(lambda x: (x[0], x[1][0]))

    # pearson: (count, sum_rating_i, sum_rating_j, sum_rating_i_sqr, sum__rating_j_sqr, sum__rating_i__rating_j)
    pearson_rdd = filtered_corated_business.aggregateByKey((0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                                           lambda a, b: (a[0] + 1, a[1] + b[0], a[2] + b[1], a[3] + b[0]**2, a[4] + b[1]**2, a[5] + b[0] * b[1]),
                                                           lambda a, b: tuple(x + y for x, y in zip(a, b)),
                                                           numPartitions = 16)
    # ((business_i, business_j), PearsonSimilarity)
    similarity = pearson_rdd.map(lambda k_v: get_PearsonSimilarity(k_v, business_avg_broadcast.value, global_avg_broadcast.value))

    # top-35 neighbors
    cf_similarity_rdd = similarity.flatMap(lambda x: [(x[0][0], (x[0][1], x[1])), (x[0][1], (x[0][0], x[1]))])
    def combine_neighbor(list1, list2):
        merge_list = list1 + list2
        # sorted by similarity (s[1]) descending
        merge_list.sort(key = lambda s: -s[1])
        return merge_list[:35]
    cf_similarity_rdd = cf_similarity_rdd.aggregateByKey([], lambda list, neighbor: (list + [neighbor])[-35:], combine_neighbor).persist(StorageLevel.MEMORY_AND_DISK)

    # prediction
    # leftOuterJoin: (user_id, (business_i, [user_id rating records] or None)) => (business_i, (user_id, [user_id rating records] or None))
    user_business_rdd = test_rdd.leftOuterJoin(user_rating_rdd)
    # cold-start problem (new users): (user_id, (business_i, None))
        # use business_avg (or global_avg if no business_avg) as predicted rating
    new_user_predict = user_business_rdd.filter(lambda x: x[1][1] is None).map(lambda x: (x[0], x[1][0], business_avg_broadcast.value.get(x[1][0], global_avg_broadcast.value)))
    # current users: (user_id, (business_i, [(business_j, stars_j), (business_k, stars_k), ......]))
        # (business_i, (user_id, [(business_j, stars_j), (business_k, stars_k), ......]))
    business_user_rdd = user_business_rdd.filter(lambda x: x[1][1] is not None).map(lambda x: (x[1][0], (x[0], x[1][1])))   
    # leftOuterJoin (get neighbors): (business_i, ((user_id, [user_id rating records]), ([(business_j, similarity_ij) neighbor list] or None)))
    join_rdd = business_user_rdd.leftOuterJoin(cf_similarity_rdd)
    user_rating_rdd.unpersist()

    rating_predict = join_rdd.map(lambda x: predict_rating(x, business_avg_broadcast.value, global_avg_broadcast.value))
    # fallback: new user predictions
    all_rating_predict = rating_predict.union(new_user_predict)
    # (user_id, business_id, predicted_rating) => ((user_id, business_id), predicted_rating) => {(user_id, business_id): predicted_rating}
    cf_prediction_dict = all_rating_predict.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

    # neighbor_cnt
    neighbor_cnt_dict = cf_similarity_rdd.mapValues(lambda lst: len(lst)).collectAsMap()
    cf_similarity_rdd.unpersist()



    ## 2. model: XGBRegressor
    
    ## select other features 

    # user.json
    user_features_rdd = sc.textFile(user_json_path).map(json.loads)

    def count_list_field(s):
        # "a,b,c" => 3; None / "None" / "" => 0
        if not s or s == "None":
            return 0
        return len(str(s).split(","))

    def get_log1p(x):
        try:
            x = float(x)
        except Exception:
            return 0.0
        if x < 0:
            return 0.0
        return math.log1p(x)

    def get_user_features(user):
        try:
            elite = user.get("elite", "None")
            elite_len = count_list_field(elite)
            friends = user.get("friends", "None")
            friends_cnt = count_list_field(friends)
            return (user["user_id"], {"u_avg_stars": float(user.get("average_stars", float("nan"))),
                                      "u_review_cnt": int(user.get("review_count", 0)),
                                      "u_fans": int(user.get("fans", 0)),
                                      "u_useful": int(user.get("useful", 0)),
                                      "u_cool": int(user.get("cool", 0)),
                                      "u_funny": int(user.get("funny", 0)),
                                      "u_friends_cnt": friends_cnt,
                                      "u_elite_years": elite_len,
                                      "u_cmp_hot": int(user.get("compliment_hot", 0)),
                                      "u_cmp_cool": int(user.get("compliment_cool", 0)),
                                      "u_cmp_funny": int(user.get("compliment_funny", 0)),
                                      "u_cmp_photos": int(user.get("compliment_photos", 0))})
        
        except Exception as e:
            return (user.get("user_id", "missing"), {})

    user_feature = user_features_rdd.map(get_user_features).collectAsMap()
    user_feature_broadcast = sc.broadcast(user_feature)

    # business.json
    business_features_rdd = sc.textFile(business_json_path).map(json.loads)

    def get_bool(attribute_dict, key):
        # convert True/False/None to 1/0
        if not isinstance(attribute_dict, dict):
            return 0
        v = attribute_dict.get(key)
        if v is None:
            return 0
        s = str(v).strip().lower()
        return 1 if s in ["true", "1", "yes"] else 0

    def get_price(attribute_dict):
        # 1~4, missing value = 2
        if not isinstance(attribute_dict, dict):
            return 2
        v = attribute_dict.get("RestaurantsPriceRange2")
        try:
            return int(v)
        except Exception:
            return 2

    def get_business_features(business):
        try:
            attrs = business.get("attributes", {}) or {}
            categories = business.get("categories", "") or ""
            is_restaurant = 1 if "Restaurant" in categories else 0
            return (business["business_id"], {"b_avg_stars": float(business.get("stars", float('nan'))),
                                              "b_review_cnt": int(business.get("review_count", 0)),
                                              "b_latitude": float(business.get("latitude", 0.0)),
                                              "b_longitude": float(business.get("longitude", 0.0)),
                                              "b_price": float(get_price(attrs)),
                                              "b_credit_card": float(get_bool(attrs, "BusinessAcceptsCreditCards")),
                                              "b_reservation": float(get_bool(attrs, "RestaurantsReservations")),
                                              "b_table_service": float(get_bool(attrs, "RestaurantsTableService")),
                                              "b_wheelchair": float(get_bool(attrs, "WheelchairAccessible")),
                                              "b_is_restaurant": float(is_restaurant)})
        
        except Exception as e:
            return (None, {})

    business_feature = business_features_rdd.map(get_business_features).collectAsMap()
    business_feature_broadcast = sc.broadcast(business_feature)

    def get_features(user, business):
        b_feature_broadcast = business_feature_broadcast.value
        business_feature = b_feature_broadcast.get(business, {})
        u_feature_broadcast = user_feature_broadcast.value
        user_feature = u_feature_broadcast.get(user, {})

        # feature list
        features = [
            # business side
            float(business_feature.get("b_avg_stars", 0.0)),
            get_log1p(business_feature.get("b_review_cnt", 0)),
            float(business_feature.get("b_latitude", 0.0)),
            float(business_feature.get("b_longitude", 0.0)),
            float(business_feature.get("b_price", 0.0)),
            float(business_feature.get("b_credit_card", 0.0)),
            float(business_feature.get("b_reservation", 0.0)),
            float(business_feature.get("b_table_service", 0.0)),
            float(business_feature.get("b_wheelchair", 0.0)),
            float(business_feature.get("b_is_restaurant", 0.0)),
            # user side
            float(user_feature.get("u_avg_stars", 0.0)),
            get_log1p(user_feature.get("u_review_cnt", 0)),
            get_log1p(user_feature.get("u_fans", 0)),
            get_log1p(user_feature.get("u_useful", 0)),
            get_log1p(user_feature.get("u_cool", 0)),
            get_log1p(user_feature.get("u_funny", 0)),
            get_log1p(user_feature.get("u_friends_cnt", 0)),
            get_log1p(user_feature.get("u_elite_years", 0)),
            get_log1p(user_feature.get("u_cmp_hot", 0)),
            get_log1p(user_feature.get("u_cmp_cool", 0)),
            get_log1p(user_feature.get("u_cmp_funny", 0)),
            get_log1p(user_feature.get("u_cmp_photos", 0))
            ]
        features = [0.0 if (not math.isfinite(f)) else f for f in features]
        
        return features

    # (user_id, business_id, stars) => (feature_list, stars)
    train_features = yelp_rdd.map(lambda x: (get_features(x[0], x[1]), x[2])).persist(StorageLevel.MEMORY_AND_DISK)
    train_data = train_features.collect()
    train_features.unpersist()
    # numpy array for XGBoost
    X_train = np.array([x for x, _ in train_data], dtype = float)
    y_train = np.array([y for _, y in train_data], dtype = float)



    ## XGBRegressor
    xgb_model = xgb.XGBRegressor(objective = "reg:linear", base_score = global_avg, reg_alpha = 0.05, reg_lambda = 1.0, n_estimators = 1000, max_depth = 5, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.7, min_child_weight = 4, nthread = 4, seed = 42)
    xgb_model.fit(X_train, y_train)
    # (user_id, business_id) => [feature_list]
    x_test = test_rdd.map(lambda x: get_features(x[0], x[1])).collect()
    x_test = np.array(x_test, dtype = float)
    # (user_id, business_id)
    test_pair_list = test_rdd.collect()

    # predict
    y_predict_xgb = xgb_model.predict(x_test)
    # (user_id, business_id, prediction)
    prediction_dict = {}
    # zip() = [((user_i, business_i), predict_i), ((user_j, business_j), predict_j), ......]
    for (user, business), y_predict in zip(test_pair_list, y_predict_xgb):
        # fallback for NaN or missing
        if y_predict != y_predict:    # check NaN
            business_stars = business_feature_broadcast.value.get(business, {}).get("b_avg_stars", global_avg)
            user_stars = user_feature_broadcast.value.get(user, {}).get("u_avg_stars", global_avg)
            business_score_fallback = 0.7 * business_avg.get(business, global_avg) + 0.3 * business_stars
            user_score_fallback = 0.7 * user_avg.get(user, global_avg) + 0.3 * user_stars
            y_predict = 0.5 * (business_score_fallback + user_score_fallback)
        # smoothing
        y_predict = 0.97 * y_predict + 0.03 * global_avg
        # clipping: reduce RMSE
        y_predict = max(1.0, min(5.0, y_predict))
        prediction_dict[(user, business)] = y_predict



    def cf_weight_from_neighbors(neighbor_cnt):
        p = math.log1p(neighbor_cnt)
        return 0.70 + 0.24 * (p / (p + 3.0))
    # prediction for (user_id, business_id)
    hybrid_predictions = []
    for (user, business) in test_pair_list:
        score_model = prediction_dict.get((user, business), global_avg)
        n_cnt = neighbor_cnt_dict.get(business, 0)
        # add CF if neighbors >= 25
        if n_cnt >= 25 and (user, business) in cf_prediction_dict:
            cf_base = 0.5 * (business_avg.get(business, global_avg) + user_avg.get(user, global_avg))
            cf_weight = cf_weight_from_neighbors(n_cnt)
            score_cf = cf_weight * cf_prediction_dict.get((user, business), cf_base) + (1 - cf_weight) * cf_base
            # n_cnt (neighbors) ↑, alpha ↑, max alpha = 0.2
            alpha = 0.1 + 0.1 * (math.log1p(n_cnt) / (math.log1p(n_cnt) + 2.0))
            score_hybrid = alpha * score_cf + (1 - alpha) * score_model
        else:
            # XGB only if neighbors < 25
            score_hybrid = score_model
        # smoothing
        score_hybrid = 0.98 * score_hybrid + 0.02 * global_avg
        # clipping
        score_hybrid = max(1.0, min(5.0, score_hybrid))
        hybrid_predictions.append((user, business, score_hybrid))

    business_avg_broadcast.unpersist()



    ## output

    with open(output_file_path, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for user, business, score_hybrid in hybrid_predictions:
            f.write(f"{user},{business},{score_hybrid}\n")

    business_feature_broadcast.unpersist()
    user_feature_broadcast.unpersist()
    global_avg_broadcast.unpersist()

    sc.stop()
    