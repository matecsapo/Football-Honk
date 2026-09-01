import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import streamlit as st

# Add source folder to path to import honk config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from honk.config import models, flagship_models

# --- APP CONFIG & THEME ---
st.set_page_config(page_title="Honk Projections", layout="wide", page_icon="🪿")

st.markdown("""
    <style>
           .block-container { padding-top: 2rem !important; }
           h2 { margin-top: 0; margin-bottom: 5px !important; }
           .down-shift { margin-top: 35px; }
           
           .as-of-date { 
                font-size: 12px; 
                color: #888; 
                margin-bottom: 8px; 
                font-weight: 500;
           }
           .highlight-white { color: white; font-weight: 700; }

           .match-strip { 
                display: flex; justify-content: space-between; align-items: center; 
                padding: 5px 0; width: 100%;
           }
           .team-block { flex: 1; display: flex; flex-direction: column; }
           .team-right { align-items: flex-end; text-align: right; }
           
           .date-block {
                flex: 0.5; display: flex; flex-direction: column; align-items: center;
                border-left: 1px solid #333; border-right: 1px solid #333;
                margin: 0 10px; min-width: 75px;
           }
           .date-sub { font-size: 10px; color: #888; text-transform: uppercase; font-weight: 700; }
           .time-main { font-size: 14px; font-weight: 900; color: white; }
           
           .loc-tag { font-size: 9px; font-weight: 800; color: #666; text-transform: uppercase; }
           .team-name { font-size: 15px; font-weight: 700; color: white; }
           .xg-id-label { font-size: 8px; color: #00FF41; font-weight: 800; text-transform: uppercase; margin-top: 5px; }
           .xg-value {
                font-size: 18px; font-weight: 900; color: white; background: #111;
                padding: 2px 8px; border-radius: 4px; border: 1px solid #333; width: fit-content;
           }
           
           .prob-bar-container {
                display: flex; width: 100%; height: 5px; border-radius: 2px; 
                overflow: hidden; margin-top: 12px;
           }
    </style>
    """, unsafe_allow_html=True)

PROJECTIONS_DIR = Path(__file__).parent / "live/projections"

# Helper to locate exact model folder on disk (checks both model_name_projection and model_name)
def resolve_model_dir(model_name):
    candidate = PROJECTIONS_DIR / f"{model_name}_projection"
    if candidate.exists():
        return candidate
    candidate_raw = PROJECTIONS_DIR / model_name
    if candidate_raw.exists():
        return candidate_raw
    return None

# Helper to robustly pull string representation from a League object or string
def to_league_str(obj):
    if hasattr(obj, "league"):
        return obj.league
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)

# Helper to look up flagship model name for a given league string
def get_flagship_model_name(league_str):
    for k, model_name in flagship_models.items():
        if to_league_str(k) == league_str:
            return model_name
    return None

# Check active status in config.models
def is_model_active(model_name):
    if model_name in models:
        return models[model_name][1]
    return False

# --- DATA HELPERS ---
def get_projection_timestamp(league, model_name):
    base = resolve_model_dir(model_name)
    if not base:
        return None
    for pattern in [f"{league}_projection_identification.json", "projection_identification.json"]:
        f = base / pattern
        if f.exists():
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    dt = datetime.fromisoformat(data['Timestamp'])
                    return dt.strftime("%b %d, %H:%M")
            except Exception:
                pass
    return None

def load_combined_standings(league, model_name):
    base = resolve_model_dir(model_name)
    if not base:
        return None
    
    exp_f = base / f"{league}_expectation/expected_standings.csv"
    if not exp_f.exists():
        exp_f = base / "expected_standings.csv"
    if not exp_f.exists(): 
        return None

    df = pd.read_csv(exp_f)
    
    mc_f = base / f"{league}_monte-carlo-simulation/monte-carlo-results.csv"
    if not mc_f.exists():
        mc_f = base / "monte-carlo-results.csv"
        
    if mc_f.exists():
        df_mc = pd.read_csv(mc_f)
        cols = ['Team', 'Avg Position', 'Title', 'CL', 'EL', 'UECL', 'Relegation']
        df = pd.merge(df, df_mc[[c for c in cols if c in df_mc.columns]], on='Team', how='left')
    return df

def load_predictions(league, model_name):
    base = resolve_model_dir(model_name)
    if not base:
        return pd.DataFrame()
        
    possible_files = [
        base / "remaining_game_predictions.csv",
        base / "remaining_games_predictions.csv",
        base / f"{league}_game_predictions.csv",
        base / "game_predictions.csv"
    ]
    for f in possible_files:
        if f.exists():
            df = pd.read_csv(f)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')
    return pd.DataFrame()

# --- STATE MANAGEMENT ---
if 'view_idx' not in st.session_state:
    st.session_state.view_idx = 0

def reset_all():
    st.session_state.t1 = None
    st.session_state.t2 = None
    st.session_state.view_idx = 0

# --- HEADER & LEAGUE SELECTOR ---
h_col, s_col = st.columns([2.2, 1], vertical_alignment="top")
with h_col:
    st.markdown("<h2>Football-Honk Projections</h2>", unsafe_allow_html=True)

# Build supported active leagues dynamically
active_leagues = set()
for key, model_name in flagship_models.items():
    league_str = to_league_str(key)
    if is_model_active(model_name):
        model_dir = resolve_model_dir(model_name)
        if model_dir:
            active_leagues.add(league_str)

leagues = sorted(list(active_leagues))

with s_col:
    st.markdown('<div class="down-shift">', unsafe_allow_html=True)
    if leagues:
        sel_league = st.selectbox("League", leagues, label_visibility="collapsed")
    else:
        sel_league = None
        st.warning("No active projection folders found in `live/projections`.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN VIEW ---
if sel_league:
    flagship_model = get_flagship_model_name(sel_league)
    main_col, feed_col = st.columns([1.8, 1])
    
    standings = load_combined_standings(sel_league, flagship_model) if flagship_model else None
    preds = load_predictions(sel_league, flagship_model) if flagship_model else pd.DataFrame()

    with main_col:
        ts = get_projection_timestamp(sel_league, flagship_model) if flagship_model else None
        if ts:
            st.markdown(f'<div class="as-of-date">Projections as of <span class="highlight-white">{ts}</span></div>', unsafe_allow_html=True)

        if standings is not None:
            mc_odds = ['Title', 'CL', 'EL', 'UECL', 'Relegation']
            metrics = ['xPts', 'Avg Position', 'xGD']
            cols = ['Team'] + [c for c in mc_odds + metrics if c in standings.columns]
            
            st_df = standings[cols].style.format({
                **{c: "{:.1%}" for c in mc_odds}, 
                'xPts': '{:.1f}', 'xGD': '{:.2f}', 'Avg Position': '{:.1f}'
            }, na_rep="-")
            
            for c, cmap in [('Title','Greens'), ('xPts','Greens'), ('Avg Position','Greens_r'), ('Relegation','Reds')]:
                if c in cols: st_df = st_df.background_gradient(subset=[c], cmap=cmap)
            
            st.dataframe(st_df, use_container_width=True, hide_index=True, height=750)
        else:
            st.info(f"No standings data found for {sel_league} inside `live/projections/{flagship_model}_projection`.")

    with feed_col:
        st.markdown("#### Upcoming Games")
        if not preds.empty:
            teams = sorted(list(set(preds['home_team'].tolist() + preds['away_team'].tolist())))
            c1, c2 = st.columns(2)
            
            t1 = c1.selectbox("T1", teams, key="t1", index=None, placeholder="Choose Team 1", label_visibility="collapsed", on_change=lambda: st.session_state.update({"view_idx": 0}))
            t2 = c2.selectbox("T2", teams, key="t2", index=None, placeholder="Choose Team 2", label_visibility="collapsed", on_change=lambda: st.session_state.update({"view_idx": 0}))
            
            filtered = preds.copy()
            if t1: filtered = filtered[(filtered['home_team'] == t1) | (filtered['away_team'] == t1)]
            if t2: filtered = filtered[(filtered['home_team'] == t2) | (filtered['away_team'] == t2)]
            
            btn_prev, btn_res, btn_next = st.columns([1, 2, 1])
            if btn_prev.button("🔼", use_container_width=True) and st.session_state.view_idx > 0:
                st.session_state.view_idx -= 1
            if btn_res.button("Reset", use_container_width=True, on_click=reset_all):
                st.rerun()
            if btn_next.button("🔽", use_container_width=True) and st.session_state.view_idx < len(filtered) - 3:
                st.session_state.view_idx += 1
            
            display_games = filtered.iloc[st.session_state.view_idx : st.session_state.view_idx + 3]
            for _, row in display_games.iterrows():
                with st.container(border=True):
                    st.markdown(f"""
                        <div class="match-strip">
                            <div class="team-block">
                                <span class="loc-tag">Home</span><span class="team-name">{row['home_team']}</span>
                                <div class="xg-id-label">Project/xG</div><div class="xg-value">{row['home_pred_goals']:.2f}</div>
                            </div>
                            <div class="date-block">
                                <span class="date-sub">{row['date'].strftime('%b %d')}</span>
                                <span class="time-main">{row['date'].strftime('%H:%M')}</span>
                            </div>
                            <div class="team-block team-right">
                                <span class="loc-tag">Away</span><span class="team-name">{row['away_team']}</span>
                                <div class="xg-id-label">Project/xG</div><div class="xg-value">{row['away_pred_goals']:.2f}</div>
                            </div>
                        </div>
                        <div class="prob-bar-container">
                            <div style="width: {row['prob_home_win']*100}%; background: #2e7d32;"></div>
                            <div style="width: {row['prob_draw']*100}%; background: #757575;"></div>
                            <div style="width: {row['prob_away_win']*100}%; background: #d32f2f;"></div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"{row['home_team']}: {row['prob_home_win']:.0%} | Draw: {row['prob_draw']:.0%} | {row['away_team']}: {row['prob_away_win']:.0%}")
        else:
            st.info("No prediction data found.")