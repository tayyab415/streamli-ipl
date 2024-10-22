import streamlit as st
import json

import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import mplsoccer
from mplsoccer import Radar, grid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from mplsoccer import VerticalPitch
import pandas as pd
import numpy as np
import mplsoccer
from mplsoccer import Radar, FontManager, grid
import matplotlib.pyplot as plt
import streamlit as st

st.title("IPL Batsmen Analysis Tool 📊")
st.subheader('Analyse your favourtite IPL teams and players\n Compare your rival players, find similar players to your favourite ones or analyse recent performance of your favourite team!')


@st.cache_data
def load_data():
    # Load your dataset here
    df = pd.read_csv(
        'players-dataset2.csv')
    return df


def load_team():
    df = pd.read_csv(
        'team-dataset.csv')
    return df


def load_data1():
    # Load the dataset
    df = pd.read_csv(
        'players-dataset.csv')

    # Replace NaN values with 0
    df.fillna(0, inplace=True)

    # Remove '%' symbol from percentage columns and convert to numeric values
    percentage_cols = [col for col in df.columns if df[col].astype(
        str).str.contains('%').any()]
    for col in percentage_cols:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with 0 (or use another appropriate fill method)
    df.fillna(0, inplace=True)

    # Select the numerical columns for generating vector embeddings
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Unnamed: 0']

    # Standardize the numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Generate vector embeddings (as numpy array)
    vector_embeddings = df[numerical_cols].values

    # Create a similarity matrix based on cosine similarity
    similarity_matrix = cosine_similarity(vector_embeddings)

    # Strip leading/trailing spaces from player names
    df['striker'] = df['striker'].str.strip()
    df = df.drop_duplicates()

    # Convert the numpy array to a DataFrame for easier handling
    similarity_df = pd.DataFrame(
        similarity_matrix, index=df['striker'], columns=df['striker']).drop_duplicates()
    similarity_df = similarity_df.drop_duplicates()
    return df, similarity_df


def get_similar_players(df, similarity_df, player_name, top_n=10):
    # Get the similarity scores for the given player
    similarity_scores = similarity_df[player_name]
    similarity_scores = similarity_scores.drop_duplicates()
    # Sort the scores in descending order and take the top_n players
    most_similar_players = similarity_scores.drop_duplicates().sort_values(
        ascending=False).head(top_n + 1)
    # Exclude the player themselves from the list
    most_similar_players = most_similar_players[most_similar_players.index != player_name]
    # Create a DataFrame with names and positions of the similar players
    similar_players_df = pd.DataFrame(most_similar_players).drop_duplicates()
    similar_players_df = similar_players_df.drop_duplicates()
    # Rename the columns
    similar_players_df.columns = ['Similarity']
    return similar_players_df


def compare_radar(pl, player1, team1, player2, team2):

    # parameter names of the statistics we want to show
    params = ['strike_rate', 'average',
              'strike_rate_powerplay', 'strike_rate_middle', 'strike_rate_death',
              'average_powerplay', 'average_middle', 'average_death', 'momentum']

# The lower and upper boundaries for the statistics
    low = [pl.strike_rate.quantile(0.05), pl.average.quantile(0.05), pl.strike_rate_powerplay.quantile(0.05), pl.strike_rate_middle.quantile(0.05), pl.strike_rate_death.quantile(
        0.05), pl.average_powerplay.quantile(0.05), pl.average_middle.quantile(0.05), pl.average_death.quantile(0.05), pl.momentum.quantile(0.05)]
    high = [pl.strike_rate.quantile(0.95), pl.average.quantile(0.95), pl.strike_rate_powerplay.quantile(0.95), pl.strike_rate_middle.quantile(0.95), pl.strike_rate_death.quantile(
        0.95), pl.average_powerplay.quantile(0.95), pl.average_middle.quantile(0.95), pl.average_death.quantile(0.95), pl.momentum.quantile(0.95)]

    from mplsoccer import Radar
    radar = Radar(params, low, high,

                  # whether to round any of the labels to integers instead of decimal places
                  round_int=[False]*len(params),
                  # the number of concentric circles (excluding center circle)
                  num_rings=4,
                  # if the ring_width is more than the center_circle_radius then
                  # the center circle radius will be wider than the width of the concentric circles
                  ring_width=1, center_circle_radius=1)

    values1 = pl.loc[pl['striker'] == player1,
                     params].drop_duplicates().fillna(0).values.flatten().tolist()
    values2 = pl.loc[pl['striker'] == player2,
                     params].drop_duplicates().fillna(0).values.flatten().tolist()


# creating the figure using the grid function from mplsoccer:
    fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                    title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot the radar
    radar.setup_axis(ax=axs['radar'], facecolor='None')
    rings_inner = radar.draw_circles(
        ax=axs['radar'], facecolor='#28252c', edgecolor='#39353f', lw=1.5)
    radar_output = radar.draw_radar(values1, ax=axs['radar'],
                                    kwargs_radar={
                                        'facecolor': 'red', 'alpha': 0.85},
                                    kwargs_rings={'facecolor': 'red', 'alpha': 0.85}) + radar.draw_radar(values2, ax=axs['radar'],
                                                                                                         kwargs_radar={
                                                                                                             'facecolor': 'blue', 'alpha': 0.6},
                                                                                                         kwargs_rings={'facecolor': 'blue', 'alpha': 0.6})


# radar_poly, rings_outer, vertices = radar_output
    range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=25, color='#fcfcfc',
                                           font='raleway')
    param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, color='#fcfcfc',
                                           font='raleway')

# adding the endnote and title text (these axes range from 0-1, i.e. 0, 0 is the bottom left)
# Note we are slightly offsetting the text from the edges by 0.01 (1%, e.g. 0.99)
    endnote_text = axs['endnote'].text(1., 0.5, 'made by Tayyab | Inspired By: StatsBomb / Rami Moghadam',
                                       color='#fcfcfc',  font='raleway',
                                       fontsize=15, ha='right', va='center')
    title1_text = axs['title'].text(0.01, 0.65, player1, fontsize=25,
                                    font='raleway', fontweight='bold',
                                    ha='left', va='center', color='red')
    title2_text = axs['title'].text(0.01, 0.25, team1, fontsize=25, fontweight='bold',
                                    font='raleway',
                                    ha='left', va='center', color='grey')
    title2_text = axs['title'].text(0.88, 0.65, player2, fontsize=25,
                                    font='raleway', fontweight='bold',
                                    ha='left', va='center', color='blue')
    title2_text = axs['title'].text(0.88, 0.25, team2, fontsize=25, fontweight='bold',
                                    font='raleway',
                                    ha='left', va='center', color='grey')

    fig.set_facecolor('black')
    return fig


def filter_data(df: pd.DataFrame, team: str, player: str):

    if team:
        df = df[df['batting_team'] == team]
    if player:
        df = df[df['striker'] == player]
    return df


def runs_bar_chart(t3: pd.DataFrame, team_t3, t31: pd.DataFrame):
    # Filter the dataset for the selected team
    filtered_data = t3[t3['batting_team'] == team_t3]

    # Get the top 10 players based on total runs
    filtered_data['total_runs'] = (filtered_data['runspowerplay'] +
                                   filtered_data['runsmiddle'] +
                                   filtered_data['runsdeath'])
    top_players = filtered_data.nlargest(10, 'total_runs')

    # Prepare data for the stacked bar chart
    chart_data = pd.DataFrame({
        'striker': top_players['striker'],
        'Runs Scored in Powerplay': top_players['runspowerplay'],
        'Runs Scored in Middle Overs': top_players['runsmiddle'],
        'Runs Scored in Death Overs': top_players['runsdeath']
    })

    # Set the index to striker for the bar chart
    chart_data.set_index('striker', inplace=True)

    # Create the first bar chart for runs
    st.subheader("Top 10 Players' Runs")
    st.bar_chart(chart_data, use_container_width=True)

    # Filter the second dataset for the selected team
    filtered_data_strike_rate = t31[t31['batting_team'] == team_t3]

    # Prepare data for the strike rate bar chart using actual values
    strike_rate_data = {
        'Strike Rate in Powerplay Overs': [filtered_data_strike_rate['strike_rate_powerplay'].values[0]],
        'Strike Rate in Middle Overs': [filtered_data_strike_rate['strike_rate_middle'].values[0]],
        'Strike Rate in Death Overs': [filtered_data_strike_rate['strike_rate_death'].values[0]]
    }

    # Create a DataFrame for plotting
    strike_rate_df = pd.DataFrame(strike_rate_data, index=[team_t3])

    # Create the second bar chart for strike rates
    st.subheader("Strike Rates")
    fig, ax = plt.subplots(figsize=(8, 5))
    strike_rate_df.T.plot(kind='bar', ax=ax, color='#BFF823')

    # Set background color
    ax.set_facecolor('#080836')
    fig.patch.set_facecolor('#080836')

    # Hide grid lines
    ax.grid(False)
    ax.legend().set_visible(False)
    # Set labels and title
    ax.set_ylabel('Strike Rate', fontsize=12, fontweight='bold', color='white')
    ax.set_title(f'Strike Rates for {team_t3}',
                 fontsize=25, fontweight='bold', color='white')

    # Set x-axis labels to horizontal and customize font
    ax.set_xticklabels(strike_rate_df.columns, rotation=0, fontsize=7,
                       fontname='Raleway', fontweight='bold', color='white')

    # Display values on top of the bars
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=10,
                     fontname='Raleway', fontweight='semibold', color='white')

    # Set y-axis limit to 200
    ax.set_ylim(0, 200)

    # Hide y-axis labels
    ax.yaxis.set_visible(False)

    st.pyplot(fig)
# Show the plot in Streamlit
# Adjust the sidebar background color and top padding


def preprocess_data(df):
    # Group by striker and aggregate the statistics
    player_stats = df.groupby('striker').agg({
        # Keep track of all teams
        'batting_team': lambda x: ', '.join(sorted(set(x))),
        'strike_rate': 'mean',
        'average': 'mean',
        'strike_rate_powerplay': 'mean',
        'strike_rate_middle': 'mean',
        'strike_rate_death': 'mean',
        'average_powerplay': 'mean', 	'average_middle': 'mean',	'average_death': 'mean',	'momentum': 'mean'
        # Add other numerical columns you want to average
        # For example:
        # 'strike_rate': 'mean',
        # 'average': 'mean',
        # etc.
    }).reset_index()

    return player_stats


tab1, tab2, tab3 = st.tabs(
    ["Find Simililar Players", "Compare Players", "Analyse Team"])

with tab1:
    st.header("IPL Batsmen Similarity Finder")

    # Load the data
    df, similarity_df = load_data1()
    df.drop_duplicates()
    similarity_df.drop_duplicates()
    similarity_df = similarity_df.loc[:, ~similarity_df.columns.duplicated()]

    # First, let the user select a team
    team0 = st.selectbox(
        "Select a team",
        df['batting_team'].sort_values().unique(),
        key="selectbox1",
        index=8
    )

    # Get unique players across all teams first
    all_players = df['striker'].unique()

    # Then filter to show only players who have played for the selected team
    team_players = df[df['batting_team'] == team0]['striker'].unique()

    # Create the player selection dropdown
    player0 = st.selectbox(
        "Select Player",
        sorted(team_players),  # Sort the players alphabetically
        key="selectbox2",
        index=None
    )

    # Select number of similar players to display
    top_n = st.slider(
        'Select number of similar players to display',
        min_value=1,
        max_value=50,
        value=10
    )

    # Button to get similar players
    if st.button('Get Similar Players'):
        if player0:
            # Get the most similar players
            similar_players = get_similar_players(
                df, similarity_df, player0, top_n)
            similar_players = similar_players.drop_duplicates()

            # Add additional information about teams for each similar player
            def get_player_teams(player):
                teams = df[df['striker'] == player]['batting_team'].unique()
                return ', '.join(teams)

            # Add a Teams column to the results
            similar_players['Teams'] = similar_players.index.map(
                get_player_teams)

            # Display the similar players with their teams
            st.write(similar_players)
        else:
            st.warning("Please select a player first")


with tab2:  # compare players
    d2 = load_data()
    pl = load_data()
    col1, col2 = st.columns(2)

    with col1:

        team1 = st.selectbox(
            "Select a team", pl['batting_team'].sort_values().unique(), key="selectbox0", index=8)
        player1 = st.selectbox(
            "Select Player 1", pl[pl['batting_team'] == team1]['striker'].sort_values().unique(), key="selectbox01", index=None)
        filtered_df1 = filter_data(pl, team1, player1)

    with col2:
        team2 = st.selectbox(
            "Select a team", pl['batting_team'].sort_values().unique(), key="selectbox3", index=9)
        player2 = st.selectbox(
            "Select Player 2", pl[pl['batting_team'] == team2]['striker'].sort_values().unique(), key="selectbox4", index=None)
        filtered_df2 = filter_data(pl, team2, player2)

    if st.button('Generate'):
        # Get the most similar players

        a = compare_radar(pl, player1, team1, player2, team2)
        a


with tab3:  # analyse team
    t3 = load_data()
    t31 = load_team()
    team_t3 = st.selectbox(
        "Select a team", pl['batting_team'].sort_values().unique(), key="selectboxt3", index=8)

    runs_bar_chart(t3, team_t3, t31)
# Display the Contact Me button
