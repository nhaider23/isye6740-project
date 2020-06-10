import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns

# grab data of career and second seasons for each player from 1990 to 2010 (who is sophomore that season)
# store in pandas: one table for career averages and one table for second season averages

# All player averages in their second season 1990-2011
#https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=single&type=per_game&per_minute_base=36&per_poss_base=100&season_start=2&season_end=2&lg_id=NBA&age_min=0&age_max=99&is_playoffs=N&height_min=0&height_max=99&year_min=1991&year_max=2011&birth_country_is=Y&as_comp=gt&as_val=0&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&order_by=season&order_by_asc=Y

#Second season averages for 2019-2020 players
#https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=single&type=per_game&per_minute_base=36&per_poss_base=100&season_start=2&season_end=2&lg_id=NBA&age_min=0&age_max=99&is_playoffs=N&height_min=0&height_max=99&year_min=2020&year_max=2020&birth_country_is=Y&as_comp=gt&as_val=0&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&order_by=season&order_by_asc=Y


#
#https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=single&type=per_game&per_minute_base=36&per_poss_base=100&season_start=2&season_end=2&lg_id=NBA&age_min=0&age_max=99&is_playoffs=N&height_min=0&height_max=99&year_min=2005&year_max=2005&birth_country_is=Y&as_comp=gt&as_val=0&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&order_by=season&order_by_asc=Y

# All Current second year players data
second_years_current_pd = pd.read_csv('data/Second_Years_2019-2020.csv')
second_years_current = second_years_current_pd.to_numpy()[1:,7:].astype(float)
second_years_current = np.nan_to_num(second_years_current)

# Dummy dataset of second year players from 2004-2005
# This needs to be career data of all players from 1990-2011 that started their second years
second_years_lebron_pd= pd.read_csv('data/Second_Years_2004-2005.csv')
second_years_lebron = second_years_lebron_pd.to_numpy()[1:,7:].astype(float)
second_years_lebron = np.nan_to_num(second_years_lebron)
print(second_years_lebron_pd.iloc[33,1])
## Missing dataset to compare career data with corresponding second year data

# PCA Dimension reduction
pca = PCA(n_components=2)
pca.fit(second_years_lebron)
#variance = pca.explained_variance_ratio_
reduced_representation_career_data = pca.transform(second_years_lebron)
print(reduced_representation_career_data[32,])

# kmedoids clustering of career stats
kmedoids = KMedoids(n_clusters=6, random_state=0).fit(reduced_representation_career_data)
kmedoids.labels_ #cluster labels
print(kmedoids.labels_ [32,])
career_centers = kmedoids.cluster_centers_ #cluster centers
print(career_centers)

# Graph clusters
y_kmedoids = kmedoids.predict(reduced_representation_career_data)
fig, ax = plt.subplots()
scatter = ax.scatter(reduced_representation_career_data[:, 0], reduced_representation_career_data[:, 1], c=y_kmedoids, s=50, cmap='viridis')
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Clusters")
ax.add_artist(legend1)
plt.scatter(career_centers[:, 0], career_centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Assign "true" cluster assignment after based on things we can look up.
# i.e. % of games started, all star appearances, all nba appearances, etc.
# Find accuracy of results
#actual_labels = 
#false_classification_rate = np.count_nonzero(kmedoids.labels_==actual_labels)/len(kmedoids.labels_)

# Map career data to second year data by row number
# Find second year "averages" for each stat of each cluster

# Fit current crop of second year players to clusters... 
# this will need to change once we implement finding second year averages per cluster
# probably euclidean distance measurements with the averages from above
pca.fit(second_years_current)
#variance = pca.explained_variance_ratio_
reduced_representation_second_years_current = pca.transform(second_years_current)
kmedoids.predict(reduced_representation_second_years_current)

