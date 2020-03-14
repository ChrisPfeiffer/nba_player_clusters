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
    st.write('running cached call')
    dataset = pd.read_csv(os.path.join("data", filename))
    dataset.drop(index=0, inplace=True)
    for c in dataset.columns:
        if dataset[c].dtype == 'O':
            dataset[c] = dataset[c].str.rstrip('%')
        
    return dataset


off_overview = get_dataset("offensive_overview.csv")
off_overview.columns = ['Player', 'Age', 'Team', 'Pos', 'Min', "Usage_rank", 'Usage', 'PSA_rank', 'PSA', 'AST_pct_rank', 'AST_pct', 'AST_usage_rank', 'AST_usg', 'TOV_pct_rank', 'TOV_pct']
convert_dict = {'AST_pct': float, 'Usage': float, 'PSA': float}
off_overview = off_overview.astype(convert_dict)
st.write(off_overview.head(3))

def_reb = get_dataset("defense_and_rebounding.csv")
def_reb.columns = ['Player', 'Age', 'Team', 'Pos', 'Min', 'BLK_pct_rank', 'BLK_pct', 'STL_pct_rank', 'STL_pct', 'FOUL%_rank', 'FOUL_pct', 'fg_OR_pct_rank', 'fg_OR_pct', 'fg_DR_pct_rank', 'fg_DR_pct', 'ftor_pct_rank', 'ftor_pct', 'ft_dr_pct_rank', 'ft_dr_pct']
convert_dict = {'BLK_pct': float, 'fg_DR_pct': float, 'fg_OR_pct': float, 'STL_pct_rank': float}
def_reb = def_reb.astype(convert_dict)
st.write(def_reb.head(3))

off_bigs_points = off_overview[off_overview.Pos.isin(["Big", "Point"])]
def_reb_bigs_points = def_reb[def_reb.Pos.isin(["Big", "Point"])]
st.write(def_reb_bigs_points.head(3))

off_and_def = off_bigs_points.merge(def_reb_bigs_points, on='Player')[['Player', 'AST_pct', 'fg_DR_pct', 'Min_x', 'Pos_x']]
off_and_def = off_and_def[off_and_def.Min_x > 480]
st.write(off_and_def.head(3))

chart = alt.Chart(off_and_def).mark_circle().encode(x='AST_pct', y='fg_DR_pct', color='Pos_x', text='Player', tooltip=['Player'])
st.altair_chart(chart, width=-1)

