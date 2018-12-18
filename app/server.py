from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from numpy import array
import implicit
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import coo_matrix


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*", "methods": "*"}})

@app.route('/recommend/<int:user_id>', methods = ['POST'])
def post(user_id):

    return jsonify(recommend(user_id, request.get_json()['reviews']))


def recommend(user_id, ranks):
    row = []
    col = []
    data = []

    print(user_id)
    print(ranks)

    max_user_idx = 0
    max_item_idx = 0
    for rank in ranks:
        if rank['star'] >= 3:
            row.append(rank['menuId'])
            col.append(rank['userId'])
            data.append(1)

        if rank['userId'] > max_user_idx: max_user_idx = rank['userId']
        if rank['menuId'] > max_item_idx: max_item_idx = rank['menuId']

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

    return recommend_list


if __name__ == '__main__':
    app.run(debug=True)