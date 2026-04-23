import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# --- APP CONFIG & THEME ---
st.set_page_config(page_title="Honk Projections", layout="wide", page_icon="🪿")

# Centralized CSS for performance and stability across devices
st.markdown("""
    <style>
           .block-container { padding-top: 2rem !important; }
           h2 { margin-top: 0; margin-bottom: 5px !important; }
           .down-shift { margin-top: 35px; }
           
           /* Timestamp styling */
           .as-of-date { 
                font-size: 12px; 
                color: #888; 
                margin-bottom: 8px; 
                font-weight: 500;
           }
           .highlight-white { color: white; font-weight: 700; }

           /* Match Strip Layout */
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

PROJECTIONS_DIR = Path(__file__).parent / "projections"

# --- DATA HELPERS ---
def get_projection_timestamp(league):
    f = PROJECTIONS_DIR / league / f"{league}_projection_identification.json"
    if f.exists():
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                dt = datetime.fromisoformat(data['Timestamp'])
                return dt.strftime("%b %d, %H:%M")
        except:
            return None
    return None

def load_combined_standings(league):
    base = PROJECTIONS_DIR / league
    exp_f = base / f"{league}_expectation/expected_standings.csv"
    mc_f = base / f"{league}_monte-carlo-simulation/monte-carlo-results.csv"
    if not exp_f.exists(): return None
    df = pd.read_csv(exp_f)
    if mc_f.exists():
        df_mc = pd.read_csv(mc_f)
        cols = ['Team', 'Avg Position', 'Title', 'CL', 'EL', 'UECL', 'Relegation']
        df = pd.merge(df, df_mc[[c for c in cols if c in df_mc.columns]], on='Team', how='left')
    return df

def load_predictions(league):
    f = PROJECTIONS_DIR / league / f"{league}_game_predictions.csv"
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
    st.markdown("<h2>🪿 Football-Honk Projections</h2", unsafe_allow_html=True)

leagues = sorted([
    d.name for d in PROJECTIONS_DIR.iterdir() 
    if d.is_dir() and (d / f"{d.name}_game_predictions.csv").exists()
])
with s_col:
    st.markdown('<div class="down-shift">', unsafe_allow_html=True)
    sel_league = st.selectbox("League", leagues, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN VIEW ---
if sel_league:
    main_col, feed_col = st.columns([1.8, 1])
    standings, preds = load_combined_standings(sel_league), load_predictions(sel_league)

    with main_col:
        # Pull and display timestamp above the table
        ts = get_projection_timestamp(sel_league)
        if ts:
            st.markdown(f'<div class="as-of-date">Projections as of <span class="highlight-white">{ts}</span></div>', unsafe_allow_html=True)

        if standings is not None:
            mc_odds = ['Title', 'CL', 'EL', 'UECL', 'Relegation']
            metrics = ['xG', 'Pts', 'Avg Position']
            cols = ['Team'] + [c for c in mc_odds + metrics if c in standings.columns]
            
            st_df = standings[cols].style.format({
                **{c: "{:.1%}" for c in mc_odds}, 
                'Pts': '{:.1f}', 'xG': '{:.2f}', 'Avg Position': '{:.1f}'
            }, na_rep="-")
            
            for c, cmap in [('Title','Greens'), ('Pts','Greens'), ('Avg Position','Greens_r'), ('Relegation','Reds')]:
                if c in cols: st_df = st_df.background_gradient(subset=[c], cmap=cmap)
            
            st.dataframe(st_df, use_container_width=True, hide_index=True, height=750)

    with feed_col:
        st.markdown("#### 🎯 Upcoming Games")
        if not preds.empty:
            # Team Selectors with Placeholder
            teams = sorted(list(set(preds['home_team'].tolist() + preds['away_team'].tolist())))
            c1, c2 = st.columns(2)
            
            t1 = c1.selectbox("T1", teams, key="t1", index=None, placeholder="Choose Team 1", label_visibility="collapsed", on_change=lambda: st.session_state.update({"view_idx": 0}))
            t2 = c2.selectbox("T2", teams, key="t2", index=None, placeholder="Choose Team 2", label_visibility="collapsed", on_change=lambda: st.session_state.update({"view_idx": 0}))
            
            filtered = preds.copy()
            if t1: filtered = filtered[(filtered['home_team'] == t1) | (filtered['away_team'] == t1)]
            if t2: filtered = filtered[(filtered['home_team'] == t2) | (filtered['away_team'] == t2)]
            
            # Navigation
            btn_prev, btn_res, btn_next = st.columns([1, 2, 1])
            if btn_prev.button("🔼", use_container_width=True) and st.session_state.view_idx > 0:
                st.session_state.view_idx -= 1
            if btn_res.button("Reset", use_container_width=True, on_click=reset_all):
                st.rerun()
            if btn_next.button("🔽", use_container_width=True) and st.session_state.view_idx < len(filtered) - 3:
                st.session_state.view_idx += 1
            
            # Display 3 matches
            display_games = filtered.iloc[st.session_state.view_idx : st.session_state.view_idx + 3]
            for _, row in display_games.iterrows():
                with st.container(border=True):
                    st.markdown(f"""
                        <div class="match-strip">
                            <div class="team-block">
                                <span class="loc-tag">Home</span><span class="team-name">{row['home_team']}</span>
                                <div class="xg-id-label">Project/xG</div><div class="xg-value">{row['home_xg']:.2f}</div>
                            </div>
                            <div class="date-block">
                                <span class="date-sub">{row['date'].strftime('%b %d')}</span>
                                <span class="time-main">{row['date'].strftime('%H:%M')}</span>
                            </div>
                            <div class="team-block team-right">
                                <span class="loc-tag">Away</span><span class="team-name">{row['away_team']}</span>
                                <div class="xg-id-label">Project/xG</div><div class="xg-value">{row['away_xg']:.2f}</div>
                            </div>
                        </div>
                        <div class="prob-bar-container">
                            <div style="width: {row['p_home']*100}%; background: #2e7d32;"></div>
                            <div style="width: {row['p_draw']*100}%; background: #757575;"></div>
                            <div style="width: {row['p_away']*100}%; background: #d32f2f;"></div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"{row['home_team']}: {row['p_home']:.0%} | Draw: {row['p_draw']:.0%} | {row['away_team']}: {row['p_away']:.0%}")