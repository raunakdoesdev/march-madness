import streamlit as st
import numpy as np
import xgboost as xgb


"# March Madness Predictor"

model = xgb.XGBClassifier()
model.load_model('xgb_model.bin')

col1, col2 = st.columns(2)

with col1:
    st.write("#### Team 1")
    team1 = [st.slider("Seed", min_value=1, max_value=64),
    st.slider("538 Rating (0-100)", min_value=0, max_value=100),
    st.slider("Win Ratio (%)", min_value=0, max_value=100) / 100.0,
    st.slider("Avg Point Delta", min_value=-100, max_value=100)]

with col2:
    st.write("#### Team 2")
    team2 = [st.slider("Seed", min_value=1, max_value=64, key='seed2'),
    st.slider("538 Rating (0-100)", min_value=0, max_value=100, key='5382'),
    st.slider("Win Ratio (%)", min_value=0, max_value=100, key='ratio2') / 100.0,
    st.slider("Avg Point Delta", min_value=-100, max_value=100, key='delta2')]



normalize = [15, 28.22, 0.46, 24.48]
feat = [((team1[0] - team2[0]) + normalize[0]) / (2 * normalize[0]),
        ((team1[1] - team2[1]) + normalize[1]) / (2 * normalize[1]),
        ((team1[2] - team2[2]) + normalize[2]) / (2 * normalize[2]),
        ((team1[3] - team2[3]) + normalize[3]) / (2 * normalize[3])]

prob_team1_wins = model.predict_proba(np.array(feat)[np.newaxis, :])[0][0]
prob_team2_wins = 1 - prob_team1_wins

st.success(f"There is a {prob_team1_wins * 100:.2f}% chance that Team 1 will win ({prob_team2_wins * 100:.2f} for Team 2)!")
st.progress(round(prob_team1_wins * 100))