from xgboost import XGBRegressor
import time, csv, json
import pandas as pd
import math
from pyspark import SparkContext
import os

def competition(folder_path, test_file_name, output_file_name):
    start_time = time.time()
    print("FOLDER PATH ", folder_path)
    print("TEST FILE NAME ", test_file_name)
    print("OUTPUT FILE NAME ", output_file_name)

    sc = SparkContext('local[*]', 'competition')
    # sc.setLogLevel('warn')
    yelp_train = sc.textFile(folder_path + "yelp_train.csv")
    yelp_val = sc.textFile(test_file_name)

    yelp_train = yelp_train.map(lambda x: x.split(","))
    yelp_val = yelp_val.map(lambda x: x.split(","))
    header = yelp_train.first()
    header_val = yelp_val.first()
    yelp_train = yelp_train.filter(lambda x: x != header)
    yelp_val = yelp_val.filter(lambda x: x != header_val)

    val_dataset = yelp_val.map(lambda x: (x[1], x[0]))

    # group by items aka businesses
    by_business = yelp_train.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(list)

    business_reviewers = yelp_train.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list).collect()
    business_reviewers_dict = {}
    for item in business_reviewers:
        business_reviewers_dict[item[0]] = item[1]

    # need a dict of business ids and dict of user ratings as the value
    by_business_temp = by_business.collect()
    dict2 = {}
    for item in by_business_temp:
        inner_dict = {}
        for user_rating_pair in item[1]:
            inner_dict[user_rating_pair[0]] = user_rating_pair[1]
        dict2[item[0]] = inner_dict

    by_user = yelp_train.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list).collect()
    by_user_dict = {}
    for item in by_user:
        by_user_dict[item[0]] = item[1]

    # get average rating
    asdf = yelp_train.map(lambda x: (1, x[2])).groupByKey().mapValues(list).collect()
    asdf = asdf[0][1]
    tot = 0
    for i in asdf:
        tot += float(i)
    avg_rating_overall = tot / len(asdf)

    # get average stars rated by business
    business_all_ratings = yelp_train.map(lambda x: (x[1], x[2])).groupByKey().mapValues(list)
    business_all_ratings = business_all_ratings.collect()
    business_avg_ratings = {}
    for item in business_all_ratings:
        ratings = [float(x) for x in item[1]]
        business_avg_ratings[item[0]] = sum(ratings) / len(ratings)

    # get average stars rated by user
    users_all_ratings = yelp_train.map(lambda x: (x[0], x[2])).groupByKey().mapValues(list)
    users_all_ratings = users_all_ratings.collect()
    user_avg_ratings = {}
    for item in users_all_ratings:
        ratings = [float(x) for x in item[1]]
        user_avg_ratings[item[0]] = sum(ratings) / len(ratings)

    # get number of reviews per business
    business_number_reviews = yelp_train.map(lambda x: (x[1], x[2])).groupByKey().mapValues(list)
    business_number_reviews = business_number_reviews.collect()
    business_number_reviews_dict = {}
    for item in business_number_reviews:
        business_number_reviews_dict[item[0]] = len(item[1])

    # number of check ins
    checkinRDD = sc.textFile(folder_path + "checkin.json")
    checkinRDD = checkinRDD.map(json.loads).map(lambda x: (x['business_id'], len(x['time'])))
    checkin = checkinRDD.collect()

    checkin_dict = {}
    for item in checkin:
        checkin_dict[item[0]] = item[1]

    tipRDD = sc.textFile(folder_path + "tip.json")
    tipRDD = tipRDD.map(json.loads).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y)
    tip = tipRDD.collect()
    tip_dict = {}
    for item in tip:
        tip_dict[item[0]] = item[1]

    def mapBusiness(x):
        try:
            goodForGroups = bool(x['attributes']['RestaurantsGoodForGroups'])
        except:
            goodForGroups = False
        try:
            priceRange = int(x['attributes']['RestaurantsPriceRange2'])
        except:
            priceRange = 1

        try:
            goodForKids = bool(x['attributes']['GoodForKids'])
        except:
            goodForKids = False

        return x['business_id'], x['stars'], x['review_count'], goodForGroups, priceRange, goodForKids

    businessRDD = sc.textFile(folder_path + "business.json").map(json.loads).map(
        lambda x: mapBusiness(x))
    businessRDD = businessRDD.collect()

    business_dict = {}
    for item in businessRDD:
        business_dict[item[0]] = [item[1], item[2], item[3], item[4], item[5]]

    base_train_file = yelp_train.collect()
    base_test_file = yelp_val.collect()

    def similarity(business1, business2):
        business1_reviewers = business_reviewers_dict[business1]
        business2_reviewers = business_reviewers_dict[business2]

        coreviewers = list(set(business1_reviewers) & set(business2_reviewers))
        # print("b1 had " + str(len(business1_reviewers)) + ", b2 had " + str(len(business2_reviewers)) +
        #      " and coreviewers have " + str(len(coreviewers)))

        # if set of users is 1 or less, correlation is 0
        if len(coreviewers) <= 1:
            return 0.25

        b1_rating_tot = 0
        b2_rating_tot = 0
        num_raters = 0

        for rater in coreviewers:
            rating_for_b1 = float(dict2[business1][rater])
            rating_for_b2 = float(dict2[business2][rater])

            b1_rating_tot += rating_for_b1
            b2_rating_tot += rating_for_b2
            num_raters += 1

        # by all ratings
        b1_rating_avg = business_avg_ratings[business1]
        b2_rating_avg = business_avg_ratings[business2]

        # by co-raters only
        # b1_rating_avg = b1_rating_tot / num_raters
        # b2_rating_avg = b2_rating_tot / num_raters

        numerator = 0
        denom_pt1 = 0
        denom_pt2 = 0

        for rater in coreviewers:
            rating_for_b1 = float(dict2[business1][rater])
            rating_for_b2 = float(dict2[business2][rater])

            numerator += (rating_for_b1 - b1_rating_avg) * (rating_for_b2 - b2_rating_avg)
            denom_pt1 += (rating_for_b1 - b1_rating_avg) ** 2
            denom_pt2 += (rating_for_b2 - b2_rating_avg) ** 2
        denom_total = math.sqrt(denom_pt1) * math.sqrt(denom_pt2)

        if denom_total == 0:
            sim = 0.25  # 1 ?
        else:
            sim = numerator / denom_total
        return sim

    # item-based CF prediction
    def get_prediction(x):
        business_to_predict = x[0]
        user_to_predict = x[1]

        # get list of businesses rated by this user
        user_rated_businesses = by_user_dict[user_to_predict]
        similarity_lst = []

        # if not in yelp_train file, just guess the overall average
        if business_to_predict not in business_reviewers_dict:
            return (avg_rating_overall, 0)

        # get similarity of businesses
        for b in user_rated_businesses:
            if b == business_to_predict:
                continue
            if b not in business_reviewers_dict:
                continue
            user_rated = dict2[b][user_to_predict]
            sim = similarity(business_to_predict, b)

            similarity_lst.append((sim, user_rated))

        similarity_lst = sorted(similarity_lst, key=lambda x: x[0], reverse=True)
        # print(similarity_lst)
        prediction_numerator = 0
        prediction_denom = 0

        neighborhood = 9999
        n_count = 0
        for item in similarity_lst:
            weight = float(item[0])
            # weight = weight * (abs(weight) ** (2.5 - 1))
            user_rating = float(item[1])

            if weight >= 0 and n_count <= neighborhood:
                prediction_numerator += weight * user_rating
                prediction_denom += abs(weight)
                n_count += 1
        if prediction_denom > 0:
            # print("Returning prediction " + str(prediction_numerator / prediction_denom))
            return (prediction_numerator / prediction_denom, n_count)
        else:
            # print("something wrong with prediction")
            return (avg_rating_overall, 0)

    final_predictions = val_dataset.map(lambda x: (x[1], x[0], get_prediction(x))).collect()
    # cf-based prediction
    # (user_id, business_id): prediction
    cf_predictions = {}
    for i in range(len(final_predictions)):
        cf_predictions[(str(final_predictions[i][0]), str(final_predictions[i][1]))] = \
            final_predictions[i][2]

    print("cf predictions done")

    # append extra features to base train file
    for item in base_train_file:
        bid = item[1]
        uid = item[0]

        # if bid in checkin_dict:
        # checkins
        # item.append(checkin_dict[bid])
        # else:
        # item.append(0)

        if bid in business_dict:
            # business avg stars
            item.append(business_dict[bid][0])
            # business review count
            item.append(business_dict[bid][1])
            # goodForGroups
            item.append(business_dict[bid][2])
            # priceRange
            item.append(business_dict[bid][3])
            # goodForKids
            item.append(business_dict[bid][4])

        else:
            item.append(3.5)
            item.append(2)
            item.append(False)
            item.append(1)
            item.append(False)

        if bid in tip_dict:
            item.append(tip_dict[bid])
        else:
            item.append(0)

        if uid in user_avg_ratings:
            item.append(user_avg_ratings[uid])
        else:
            item.append(3.5)

    base_train_file_dict = {}
    i = 0
    for item in base_train_file:
        base_train_file_dict[i] = item
        i += 1

    df = pd.DataFrame.from_dict(base_train_file_dict,
                                orient='index',
                                columns=['user_id', 'business_id', 'stars',
                                         # 'checkins',
                                         'avg_stars',
                                         'num_reviews',
                                         'goodForGroups',
                                         'priceRange',
                                         'goodForKids',
                                         'num_tips',
                                         'user_avg_ratings',
                                         ])
    X_train = df.drop(['stars', 'user_id', 'business_id'], axis=1)
    Y_train = df['stars']

    for item in base_test_file:
        bid = item[1]
        uid = item[0]
        # if bid in checkin_dict:
        # checkins
        # item.append(checkin_dict[bid])
        # else:
        # item.append(0)

        if bid in business_dict:
            # business avg stars
            item.append(business_dict[bid][0])
            # business review count
            item.append(business_dict[bid][1])
            # goodForGroups
            item.append(business_dict[bid][2])
            # priceRange
            item.append(business_dict[bid][3])
            # goodForKids
            item.append(business_dict[bid][4])
        else:
            item.append(3.5)
            item.append(2)
            item.append(False)
            item.append(1)
            item.append(False)

        if bid in tip_dict:
            item.append(tip_dict[bid])
        else:
            item.append(0)

        if uid in user_avg_ratings:
            item.append(user_avg_ratings[uid])
        else:
            item.append(3.5)

    base_test_file_dict = {}
    i = 0
    for item in base_test_file:
        base_test_file_dict[i] = item
        i += 1

    dftest = pd.DataFrame.from_dict(base_test_file_dict,
                                    orient='index',
                                    columns=['user_id', 'business_id',
                                             'stars',
                                             # 'checkins',
                                             'avg_stars',
                                             'num_reviews',
                                             'goodForGroups',
                                             'priceRange',
                                             'goodForKids',
                                             'num_tips',
                                             'user_avg_ratings',
                                             ])
    X_test = dftest.drop(['user_id',
                          'business_id',
                          'stars',
                          ], axis=1)
    # Y_test = dftest['stars']

    model = XGBRegressor()
    model.fit(X_train, Y_train)

    preds = model.predict(X_test)

    actual_stars_val = yelp_val.map(lambda x: (x[0], x[1])).collect()
    actual_stars_dict = {}
    for item in actual_stars_val:
        actual_stars_dict[(item[0], item[1])] = 1

    # model-based prediction
    # (user_id, business_id): prediction
    model_predictions = {}
    for i in range(len(preds)):
        model_predictions[(str(actual_stars_val[i][0]), str(actual_stars_val[i][1]))] = preds[i]

    print("model predictions done")

    # hybrid prediction
    hybrid_predictions = {}
    for k, v in cf_predictions.items():
        hybrid_predictions[k] = 0
    max_neighbor = -2
    for k, v in hybrid_predictions.items():
        num_neighbors = cf_predictions[k][1]
        if num_neighbors > max_neighbor:
            max_neighbor = num_neighbors
        factor = min(1.0, num_neighbors / 400)

        hybrid_predictions[k] = (1 - factor) * float(model_predictions[k]) + factor * float(cf_predictions[k][0])

    try:
        os.remove(output_file_name)
    except OSError:
        pass
    with open(output_file_name, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["user_id", "business_id", "prediction"])
        for k, v in hybrid_predictions.items():
            str_to_write = [str(k[0]), str(k[1]), str(v)]
            writer.writerow(str_to_write)

    RMSE = 0

    for k, v in hybrid_predictions.items():
       RMSE += (float(hybrid_predictions[k]) - float(actual_stars_dict[k])) ** 2

    RMSE = math.sqrt(RMSE / len(hybrid_predictions))
    print("HYBRID RMSE IS ", RMSE)

    end_time = time.time()
    print("Duration: ", end_time - start_time)
    return


if __name__ == '__main__':
    folder_path = '/Users/matthewtung/PycharmProjects/dsci553_assignment3/data/'
    test_file_name = '/Users/matthewtung/PycharmProjects/dsci553_assignment3/data/'
    output_file_name = 'output_task2_3.csv'
    # /mnt/vocwork4/ddd_v1_w_aMl_1382908/asn1085231_2/asn1085232_1/resource/asnlib/publicdata

    competition(folder_path, test_file_name, output_file_name)
