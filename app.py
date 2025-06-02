import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Gaming Analytics Dashboard", layout="wide")
st.title("🎮 Gaming Analytics Dashboard")
st.markdown("Analyze funnel drop-off, player segmentation, and retention by acquisition channel.")
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
sns.set_style("whitegrid")
sns.set_palette("pastel")  # You can also try "coolwarm" or "rocket"

@st.cache_data
def generate_data(num_players=10000):
    np.random.seed(42)
    player_data = pd.DataFrame({
        'player_id': [f'player_{i}' for i in range(1, num_players + 1)],
        'install_day': np.random.randint(1, 5, size=num_players),
        'country': np.random.choice(['US', 'IN', 'BR', 'DE', 'JP'], size=num_players, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'device': np.random.choice(['iOS', 'Android'], size=num_players, p=[0.4, 0.6]),
        'acquisition_channel': np.random.choice(['Organic', 'Facebook Ads', 'Google Ads', 'Influencer'],
                                                size=num_players, p=[0.5, 0.2, 0.2, 0.1]),
        'spend': np.round(np.random.exponential(scale=5, size=num_players), 2)
    })
    player_data['is_spender'] = (player_data['spend'] > 0).astype(int)

    funnel = pd.DataFrame({'player_id': player_data['player_id']})
    funnel['completed_tutorial'] = np.random.binomial(1, 0.98, num_players)
    funnel['completed_level_1'] = np.where(funnel['completed_tutorial'] == 1, np.random.binomial(1, 0.95, num_players), 0)
    funnel['completed_level_5'] = np.where(funnel['completed_level_1'] == 1, np.random.binomial(1, 0.65, num_players), 0)
    funnel['completed_level_10'] = np.where(funnel['completed_level_5'] == 1, np.random.binomial(1, 0.4, num_players), 0)

    behavior = pd.DataFrame({
        'player_id': player_data['player_id'],
        'sessions': np.random.poisson(10, num_players),
        'avg_session_length': np.random.normal(30, 10, num_players).clip(min=5),
        'days_active': np.random.randint(1, 31, num_players),
        'spend': player_data['spend']
    })

    X_scaled = StandardScaler().fit_transform(behavior[['sessions', 'avg_session_length', 'days_active', 'spend']])
    behavior['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)

    logs = []
    for day in range(1, 8):
        logs.append(pd.DataFrame({
            'player_id': player_data['player_id'],
            'acquisition_channel': player_data['acquisition_channel'],
            'cohort_day': day,
            'active': np.random.binomial(1, np.exp(-0.3 * day), num_players)
        }))
    cohort = pd.concat(logs)
    retention = cohort.groupby(['acquisition_channel', 'cohort_day'])['active'].mean().unstack()

    return funnel, behavior, retention

funnel_data, behavior_data, retention_data = generate_data()

tab1, tab2, tab3 = st.tabs(["📉 Funnel Drop-off", "🧠 Cluster Analysis", "📈 Retention Curves"])

with tab1:
    st.header("Player Progression Funnel")

    # Only use numeric columns for mean calculation
    funnel_means = funnel_data.select_dtypes(include='number').mean().round(2)

    # Create bar chart
    fig, ax = plt.subplots()
    sns.barplot(x=funnel_means.index, y=funnel_means.values, ax=ax)
    ax.set_title("Funnel Drop-off: Tutorial → L1 → L5 → L10")
    ax.set_ylim(0, 1)
    for i, v in enumerate(funnel_means.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

    # Display in Streamlit
    st.pyplot(fig)
    st.dataframe(funnel_means.to_frame(name="Rate"))

with tab2:
    st.header("K-Means Player Clusters")
    st.dataframe(behavior_data.groupby('cluster')[['sessions', 'avg_session_length', 'days_active', 'spend']].mean().round(2))
    fig, ax = plt.subplots()
    sns.boxplot(data=behavior_data, x='cluster', y='spend', ax=ax)
    ax.set_title("Spending Distribution by Cluster")
    st.pyplot(fig)

with tab3:
    st.header("7-Day Retention by Acquisition Channel")
    fig, ax = plt.subplots(figsize=(10, 5))
    for channel in retention_data.index:
        ax.plot(retention_data.columns, retention_data.loc[channel], label=channel)
    ax.set_title("Retention Curves (Day 1–7)")
    ax.set_xlabel("Day Since Install")
    ax.set_ylabel("Retention Rate")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(retention_data.round(2))
