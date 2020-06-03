import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# grab data of career and second seasons for each player from 1990 to 2010 (who is sophomore that season)
# store in pandas: one table for career averages and one table for second season averages

# All player averages in their second season 1990-2011
#https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=single&type=per_game&per_minute_base=36&per_poss_base=100&season_start=2&season_end=2&lg_id=NBA&age_min=0&age_max=99&is_playoffs=N&height_min=0&height_max=99&year_min=1991&year_max=2011&birth_country_is=Y&as_comp=gt&as_val=0&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&order_by=season&order_by_asc=Y

#Second season averages for 2019-2020 players
#https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=single&type=per_game&per_minute_base=36&per_poss_base=100&season_start=2&season_end=2&lg_id=NBA&age_min=0&age_max=99&is_playoffs=N&height_min=0&height_max=99&year_min=2020&year_max=2020&birth_country_is=Y&as_comp=gt&as_val=0&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&order_by=season&order_by_asc=Y

# PCA Dimension reduction
pca = PCA(n_components=2)
pca.fit(career_data)
variance = pca.explained_variance_ratio_
reduced_representation_career_data = pca.transform(data)

# kmedoids clustering
kmedoids = KMedoids(n_clusters=2, random_state=0).fit
kmedoids.labels_ #cluster labels
kmedoids.cluster_centers_ #cluster centers

# actual custer assignment
# assign each player a "true" cluster assignment after seeing the clustering by kmedoids
# perform some panda manipuation and add column with real cluster assignment
# find classification rate accuracy of our algorithm
false_classification_rate = np.count_nonzero(kmedoid_assignments==actual_clusters)/len(kmedoid_assignments)

# Map each players' data point above to their second season pandas table
# Find average of each clusters' second season performances...


# See which NBA players in their second seasons for current season are closest to above averages :) and profit
