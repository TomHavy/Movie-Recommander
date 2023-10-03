import sys

sys.path.append("C:/Users/thavyarimana/RecSys/src")

from recommendations.collab_filtering import (
    load_ratings,
    model,
    get_rating,
)

data = load_ratings()

svd = model(data)

pred_rating = get_rating(svd,user_id=1,movie_id=500)

print(pred_rating)