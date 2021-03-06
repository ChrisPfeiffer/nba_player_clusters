import streamlit as st
import matplotlib as plt
import seaborn as sns
import altair as alt
import pandas as pd
import numpy as np
import os
from os import path
from sklearn.cluster import KMeans
from sklearn import preprocessing

@st.cache
def get_dataset(filename):
    dataset = pd.read_csv(os.path.join("data", filename))
    dataset.drop(index=0, inplace=True)
    for c in dataset.columns:
        if dataset[c].dtype == 'O':
            dataset[c] = dataset[c].str.rstrip('%')
        
    return dataset

'# Finding Groups of NBA Players using K-Means Clustering'

(
    'The goal of this analysis is to use unsupervised machine learning, specifically K-Means Clustering to see'
    ' which groups of players naturally emerge and what their characteristics are. We will start off very simply by'
    ' splitting a group of only bigs and points into two groups using two data features. This allows us to easily visualize'
    ' how k-means clustering works. After that, we will use many more features to come up with 4 clusters of NBA players and inspect'
    ' the players and characteristics of each cluster. Finally, we will see clusters with completely different characteristics emerge'
    ' when we perform the same analysis on players from the 2003-2004 season.'
)

'## Importing the raw data'
off_overview = get_dataset("offensive_overview.csv")
off_columns = ['Player', 'Age', 'Team', 'Pos', 'Min', "Usage_rank", 'Usage', 'PSA_rank', 'PSA', 'AST_pct_rank', 'AST_pct', 'AST_usage_rank', 'AST_usg', 'TOV_pct_rank', 'TOV_pct']
off_overview.columns = off_columns
off_convert_dict = {'AST_pct': float, 'Usage': float, 'PSA': float}
off_overview = off_overview.astype(off_convert_dict)
st.write('All data comes from Cleaning the Glass. I scraped that data into .csv files that I import here.')
st.write('The first few rows of the offensive data:')
st.write(off_overview.head(5))

def_reb = get_dataset("defense_and_rebounding.csv")
def_columns = ['Player', 'Age', 'Team', 'Pos', 'Min', 'BLK_pct_rank', 'BLK_pct', 'STL_pct_rank', 'STL_pct', 'FOUL%_rank', 'FOUL_pct', 'fg_OR_pct_rank', 'fg_OR_pct', 'fg_DR_pct_rank', 'fg_DR_pct', 'ftor_pct_rank', 'ftor_pct', 'ft_dr_pct_rank', 'ft_dr_pct']
def_reb.columns = def_columns
def_convert_dict = {'BLK_pct': float, 'fg_DR_pct': float, 'fg_OR_pct': float, 'STL_pct_rank': float}
def_reb = def_reb.astype(def_convert_dict)
st.write('The first few rows of the defensive data:')
st.write(def_reb.head(5))

off_bigs_points = off_overview[off_overview.Pos.isin(["Big", "Point"])]
def_reb_bigs_points = def_reb[def_reb.Pos.isin(["Big", "Point"])]

off_and_def = off_bigs_points.merge(def_reb_bigs_points, on='Player')[['Player', 'AST_pct', 'fg_DR_pct', 'Min_x', 'Pos_x']]
off_and_def = off_and_def[off_and_def.Min_x > 480]
(
    'For the simplified demo of clustering, I limited the data to Points and Bigs. I am using only Assist Percentage and Defensive Rebound Percentage.'
    ' Those two features should create contrast between the points and bigs. Sample of joined and filtered data:'
)
st.write(off_and_def.head(5))

chart = alt.Chart(off_and_def).mark_circle().encode(x='AST_pct', y='fg_DR_pct', color='Pos_x', text='Player', tooltip=['Player'])
st.write('A scatter plot of the joined and filtered data. Hover over a dot to see who it is.')
st.altair_chart(chart)

st.write('Now I will perform K Means Clustering with two clusters and add the cluster each player is assigned to the data')
kmeans = KMeans(n_clusters=2, random_state=32).fit(off_and_def[['AST_pct', 'fg_DR_pct']])
clusters = kmeans.predict(off_and_def[['AST_pct', 'fg_DR_pct']])
off_and_def['cluster'] = clusters

st.write(off_and_def.head(3))

st.write("As you can see, two natural clusters emerged representing the different play styles. Cluster 0 contains almost entirely bigs while Cluster 1 contains almost entirely points despite the fact that I gave the model no information about position")

chart2 = alt.Chart(off_and_def).mark_point().encode(
    x = 'AST_pct', 
    y = 'fg_DR_pct', 
    shape = 'cluster:O',
    color = 'Pos_x',
    tooltip = ['Player']
)

st.altair_chart(chart2)


st.write("Type in any player to see which cluster they are in: ")

player = st.text_input('Player Name', 'Ben Simmons')
off_and_def[off_and_def.Player==player]

st.write("These are the bigs that got grouped with the points. All have high Assist Percentages")
st.write(off_and_def[(off_and_def.Pos_x=='Big') & (off_and_def.cluster==1)])
st.write("These are the points that got grouped with the bigs:")
st.write(off_and_def[(off_and_def.Pos_x=='Point') & (off_and_def.cluster==0)])
st.write("Pat Bev is a good rebounder. How is Collin Sexton's Assist Percentage this low? Who is Frank Jackson?")

'## K-means clustering with more features and four groups'

st.write("Now I am going to include a more comprehensive set of player attributes and see what clusters emerge")

shot_types = get_dataset("shot_types.csv")
shot_columns = ['Player', 'Age', 'Team', 'Pos', 'MIN', 'efg_rank', 'efg', 'Rim_rank', 'Rim', 'Short_mid_rank', 'Short_mid', 'Long_mid', 'Long_mid_rank', 'All_mid_rank', 'All_mid', 'Corner_three_rank', 'Corner_three', 'Non_corner_three_rank', 'Non_corner_three', 'All_three_rank', 'All_three']
shot_types.columns = shot_columns
shot_conversion_dict = {'All_mid': float, 'Rim': float, 'All_three': float}
shot_types = shot_types.astype(shot_conversion_dict)

foul_drawing = get_dataset("foul_drawing.csv")
foul_columns = ["Player", "Age", "Team", "Pos", "Min", "FT_rank", "FT_pct", 'SF_rank', 'SF_pct', 'FF_rank', 'FF_pct', 'And1_rank', 'And1_pct']
foul_drawing.columns = foul_columns
foul_conversion_dict = {"FT_pct": float, "SF_pct": float}
foul_drawing = foul_drawing.astype(foul_conversion_dict)

kmeans_data = off_overview[['Player', 'Team', 'Pos', 'Min', 'Usage', 'AST_pct', 'PSA']]
kmeans_data = kmeans_data.merge(def_reb[['Player', 'Team','BLK_pct', 'STL_pct', 'fg_OR_pct', 'fg_DR_pct']], on=['Player', 'Team'])
kmeans_data = kmeans_data.merge(foul_drawing[['Player', 'Team', 'FT_pct', 'SF_pct']], on=['Player', 'Team'])
kmeans_data = kmeans_data.merge(shot_types[['Player', 'Team', 'Rim', 'All_mid', 'All_three']], on=['Player', 'Team'])
kmeans_data = kmeans_data[kmeans_data['Min']>400]

st.write("Here is all the joined data/metrics I plan to use as input. It only contains players who have played more than 400 minutes this season.")
st.write("I am also only using statistics that quantify HOW a player plays, not HOW MUCH they play. I am looking for playing styles. For example I'm using usage rate instead of shots per game.")

training_values = kmeans_data.drop(['Player', 'Team', 'Pos', 'Min'], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
scaled_values = pd.DataFrame(min_max_scaler.fit_transform(training_values))

kmeans = KMeans(n_clusters=4, random_state=32).fit(scaled_values)
clusters = kmeans.predict(scaled_values)
kmeans_data['clusters'] = clusters

st.write("4 Natural groups emerged. Here are their average metrics by group:")
averages = kmeans_data.groupby('clusters').mean().sort_values('Usage')
averages['cluster_name'] = ['1', '2', '3', '4']
averages.loc[averages.PSA < 110, 'cluster_name'] = "Well Rounded Role Players"
cluster_wrrp = averages[averages.PSA <110].index.values[0]
averages.loc[averages.All_three > 55, 'cluster_name'] = "3pt Role Players"
cluster_3pt = averages[averages.All_three > 55].index.values[0]
averages.loc[averages.Rim > 60, 'cluster_name'] = "Traditional Bigs"
cluster_rim = averages[averages.Rim > 60].index.values[0]
averages.loc[averages.Usage > 25, 'cluster_name'] = "Offensive Focal Points"
cluster_focals = averages[averages.Usage > 25].index.values[0]
averages

'### Well Rounded Role Players'
averages[averages.PSA < 110]
'This cluster has the lowest points per shot attempt, but does not lead or come in last in any other category. This group is better at rebounding than the three point shooting role players while being less efficient offensively'
kmeans_data[kmeans_data.clusters == cluster_wrrp][['Player', 'Team', 'Pos']]
st.write(kmeans_data[kmeans_data.clusters == cluster_wrrp].groupby('Pos').count()['Player'])

'### Three Point Shooting Specialist Role Players'
averages[averages.All_three > 55]
'This Cluster takes more than half of their shots from three and the lowest percent from mid range. This is the only category they lead. They are last in usage, but second in points per shot attempt.'
kmeans_data[kmeans_data.clusters == cluster_3pt][['Player', 'Team', 'Pos']]
st.write(kmeans_data[kmeans_data.clusters == cluster_3pt].groupby('Pos').count()['Player'])

'### Traditional Bigs'
averages[averages.Rim > 60]
'This cluster has the highest Points Per Shot Attempt, Defensive Field Goal Rebounding %, and takes most of their shots near the rim. These are the old school big men'
kmeans_data[kmeans_data.clusters == cluster_rim][['Player', 'Team', 'Pos']]
st.write(kmeans_data[kmeans_data.clusters == cluster_rim].groupby('Pos').count()['Player'])
'Every Single Member of Cluster 0 is a big except Ben Simmons!'


'### Offensive focal points'
'Cluster 3 has the highest usage rate, highest assist rate, highest free throw percentage, and takes a balance of shots at the rim, mid, and three point ranges'
kmeans_data[kmeans_data.clusters == cluster_focals][['Player', 'Team', 'Pos']]
st.write(kmeans_data[kmeans_data.clusters == cluster_focals].groupby('Pos').count()['Player'])
'Nikola Jokic is the only big in this category'

'### Observations'

'It is interesting to me that the group with the lowest usage rate (1), took the most threes. In the modern NBA, so many players have a permission slip the let the three fly.'
'The Traditional Bigs Cluster has the least amount of players by far. This dwindling group of non-three-shooting bigs has managed to avoid extinction with insanely good points per shot attempt'
'The Ben Simmons shot profile looks like that of a mid-90s center, not a modern guard.'

'## Same Analysis - Different Era'

'Cleaning the Glass has all of these metrics all the way back to the 2003-2004 Season. My hypothesis is that very different groups of players will emerge.'

off_overview03 = get_dataset("offensive_overview_03.csv")
off_overview03.columns = off_columns
off_overview03 = off_overview03.astype(off_convert_dict)

def_reb03 = get_dataset("defense_and_rebounding_03.csv")
def_reb03.columns = def_columns
def_reb03 = def_reb03.astype(def_convert_dict)

shot_types03 = get_dataset("shot_types_03.csv")
shot_types03.columns = shot_columns
shot_types03 = shot_types03.astype(shot_conversion_dict)

foul_drawing03 = get_dataset("foul_drawing_03.csv")
foul_columns = ["Player", "Age", "Team", "Pos", "Min", "FT_rank", "FT_pct", 'SF_rank', 'SF_pct', 'FF_rank', 'FF_pct', 'And1_rank', 'And1_pct']
foul_drawing03.columns = foul_columns
foul_conversion_dict = {"FT_pct": float, "SF_pct": float}
foul_drawing03 = foul_drawing03.astype(foul_conversion_dict)

kmeans_data03 = off_overview03[['Player', 'Team', 'Pos', 'Min', 'Usage', 'AST_pct', 'PSA']]
kmeans_data03 = kmeans_data03.merge(def_reb03[['Player', 'Team','BLK_pct', 'STL_pct', 'fg_OR_pct', 'fg_DR_pct']], on=['Player', 'Team'])
kmeans_data03 = kmeans_data03.merge(foul_drawing03[['Player', 'Team', 'FT_pct']], on=['Player', 'Team'])
kmeans_data03 = kmeans_data03.merge(shot_types03[['Player', 'Team', 'Rim', 'All_mid', 'All_three']], on=['Player', 'Team'])
kmeans_data03 = kmeans_data03[kmeans_data03['Min']>500]
training_values03 = kmeans_data03.drop(['Player', 'Team', 'Pos', 'Min'], axis=1)
min_max_scaler03 = preprocessing.MinMaxScaler()
scaled_values03 = pd.DataFrame(min_max_scaler.fit_transform(training_values03))

kmeans03 = KMeans(n_clusters=4, random_state=32).fit(scaled_values03)
clusters03 = kmeans03.predict(scaled_values03)
kmeans_data03['clusters'] = clusters03
'Note: Cleaning the glass did not have Shooting foul percent so I removed that from 03-04 clusters'
'The four clusters from the 03-04 season:'

averages03 = kmeans_data03.groupby('clusters').mean().sort_values('Usage')
averages03
averages03['cluster_name'] = ['1', '2', '3', '4']
averages03.loc[averages03.All_mid > 60, 'cluster_name'] = "Mid Range Bigs"
cluster_mrb = averages03[averages03.All_mid >60].index.values[0]
averages03.loc[averages03.Usage > 21, 'cluster_name'] = "Offensive Focal Points"
cluster_focals = averages03[averages03.Usage > 21].index.values[0]
averages03.loc[averages03.Rim > 55, 'cluster_name'] = "Old School Bigs"
cluster_bigs = averages03[averages03.Rim > 55].index.values[0]
averages03.loc[averages03.All_three > 30, 'cluster_name'] = "Skilled Role Players"
cluster_skilled = averages03[averages03.All_three > 30].index.values[0]
'The clusters have very different looks to them'

'### Cluster 0 - Bigs who love the mid-range'
'Cluster 0 has the second highest rebounding numbers, barely shoots any threes, and shoots more midranges than any group from either era.'
'Close your eyes and picture Kevin Garnett pulling up from the elbow - that is this group'
'Most of these guys would be shooting threes as stretch fours or fives in todays game'
st.write(kmeans_data03[kmeans_data03.clusters==cluster_mrb])
st.write(kmeans_data03[kmeans_data03.clusters==cluster_mrb].groupby('Pos').count()['Player'])

'### Cluster 1 - Offensive focal points'
'Cluster 1 has the highest Usage and Ast pct and shoots half their shots from mid range'
st.write(kmeans_data03[kmeans_data03.clusters==cluster_focals])
st.write(kmeans_data03[kmeans_data03.clusters==cluster_focals].groupby('Pos').count()['Player'])


'### Cluster 2 - Old school bigs'
'Cluster 2 rarely assists and virtually never shoots threes. More than half their shots are from the inside.'
st.write(kmeans_data03[kmeans_data03.clusters==cluster_bigs])
st.write(kmeans_data03[kmeans_data03.clusters==cluster_bigs].groupby('Pos').count()['Player'])

'### Cluster 3 - Skilled Role Players'
'This is the most efficient group in terms of points per shot attempt. They shoot the most threes in this era.'
st.write(kmeans_data03[kmeans_data03.clusters==cluster_skilled])
st.write(kmeans_data03[kmeans_data03.clusters==cluster_skilled].groupby('Pos').count()['Player'])

'In 2003-2004, only one cluster took more than thirty percent of their shots from three. All except the traditional bigs took more than 30 percent of their shots from three in the modern NBAs' 
'The highest percentage of shots from mid-range in the modern NBA, is roughly equal to the lowest percentage of shots from mid range in 03-04'
'In 2003-2004, the skilled role players, the group that shot the most threes, had the most points per shot attempt. Possibly an early sign the NBA was undervaluing the three point shot in this era.'