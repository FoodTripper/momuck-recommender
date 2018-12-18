import numpy as np
from numpy import array
import implicit
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import coo_matrix

def recommend(user_id, ranks):
    row = []
    col = []
    data = []

    max_user_idx = 0
    max_item_idx = 0
    for rank in ranks:
        if rank['star'] >= 3:
            row.append(rank['menu_id'])
            col.append(rank['user_id'])
            data.append(1)

        if rank['user_id'] > max_user_idx: max_user_idx = rank['user_id']
        if rank['menu_id'] > max_item_idx: max_item_idx = rank['menu_id']

    row = array(row)
    col = array(col)
    data = array(data, dtype=np.float32)

    ranks = coo_matrix((data, (row, col)), shape=(max_item_idx + 1, max_user_idx + 1))
    user_item_data = ranks.T.tocsr()

    ranks = (bm25_weight(ranks, B=0.9) * 5).tocsr()

    model = implicit.als.AlternatingLeastSquares()

    model.fit(ranks)

    recommendations = model.recommend(user_id, user_item_data)
    recommend_list = []
    for r in recommendations:
        recommend_list.append(r[0])

    print(recommend_list)
    return recommendations
    
f = open('../restaurant.data')
lines = f.readlines()

ranks = []
for line in lines:
    user_id, item_id, score, timestamp = line.split()
    i_user_id = int(user_id)
    i_item_id = int(item_id)
    score = float(score)
    timestamp = int(timestamp)

    ranks.append({'user_id': i_user_id, 'menu_id': i_item_id, 'star': score })

recommend(1, ranks)