import pandas as pd
import pulp
import re
import requests
import unicodedata
import random
import time
import os
from flask import Flask, render_template_string, request
from datetime import datetime
import pytz
from pybaseball import batting_stats, pitching_stats, statcast_pitcher_exitvelo_barrels

app = Flask(__name__)

# --- CACHE CONTROL ---
_STATS_CACHE = {'h': None, 'p': None, 'time': 0}

# --- CONFIG ---
SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"
POS_ORDER = {'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4, '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9}

TEAM_MAP = {
    "CHW": "CWS", "CHA": "CWS", "CWS": "CWS", "WSH": "WAS", "WAS": "WAS",
    "OAK": "OAK", "ATH": "OAK", "SF": "SF", "SFO": "SF", "SFG": "SF",
    "AZ": "ARI", "ARI": "ARI", "TB": "TB", "TBA": "TB", "KC": "KC",
    "KCA": "KC", "SD": "SD", "SDN": "SD", "NYY": "NYY", "NYA": "NYY",
    "NYM": "NYM", "NYN": "NYM", "LAD": "LAD", "LAN": "LAD", "STL": "STL",
    "SLN": "STL", "CHC": "CHC", "CHN": "CHC", "TOR": "TOR", "COL": "COL", "ATL": "ATL"
}

TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
    "SEA": 136, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WAS": 120
}


def get_logo_url(team_abbr):
    clean_abbr = TEAM_MAP.get(team_abbr, team_abbr)
    tid = TEAM_ID_MAP.get(clean_abbr)
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{tid}.svg" if tid else "https://www.mlbstatic.com/team-logos/league/1.svg"


def normalize_name(name):
    if not isinstance(name, str): return ""
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = re.sub(r'[^a-zA-Z\s]', '', name).lower().strip()
    for s in [' jr', ' sr', ' iii', ' ii', ' iv']:
        if name.endswith(s): name = name[:-len(s)]
    return name.strip()


def clean_hand_str(h):
    if pd.isna(h): return "?"
    s = str(h).upper()
    return s[0] if s and s[0] in 'RLS' else "?"


def get_espn_game_times():
    now_et = datetime.now(pytz.timezone('US/Eastern'))
    today_str = now_et.strftime('%Y%m%d')
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today_str}&limit=100"
    slate_times = {}
    try:
        data = requests.get(url, timeout=5).json()
        for event in data.get('events', []):
            competitions = event.get('competitions', [{}])[0]
            competitors = competitions.get('competitors', [])
            raw_names = [TEAM_MAP.get(t['team']['abbreviation'].upper(), t['team']['abbreviation'].upper()) for t in
                         competitors]
            game_id = " vs ".join(sorted(raw_names))
            utc_time = datetime.strptime(event['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=pytz.utc)
            et_dt = utc_time.astimezone(pytz.timezone('US/Eastern'))
            slate_times[game_id] = {'display': et_dt.strftime('%I:%M %p'), 'raw': et_dt}
    except:
        pass
    return slate_times


def get_weighted_stats():
    global _STATS_CACHE
    if _STATS_CACHE['h'] is not None and (time.time() - _STATS_CACHE['time']) < 600:
        return _STATS_CACHE['h'], _STATS_CACHE['p']
    try:
        h26, h25 = batting_stats(2026), batting_stats(2025)
        h_merge = pd.merge(h26, h25, on='Name', how='outer', suffixes=('_26', '_25')).fillna(0)

        # Ensure columns exist before weight calc
        for col in ['Barrel%_26', 'Barrel%_25', 'xwOBA_26', 'xwOBA_25']:
            if col not in h_merge.columns: h_merge[col] = 0

        h_merge['W_Barrel'] = (h_merge['Barrel%_26'] * 0.7) + (h_merge['Barrel%_25'] * 0.3)
        h_merge['W_xwOBA'] = (h_merge['xwOBA_26'] * 0.7) + (h_merge['xwOBA_25'] * 0.3)
        h_merge['Edge_Value'] = (h_merge['W_Barrel'] * 50) + (h_merge['W_xwOBA'] * 20)
        h_merge['norm_name'] = h_merge['Name'].apply(normalize_name)
        h_final = h_merge[['norm_name', 'Edge_Value', 'W_Barrel', 'W_xwOBA']].rename(
            columns={'W_Barrel': 'Barrel%', 'W_xwOBA': 'xwOBA'})

        raw_barrels = statcast_pitcher_exitvelo_barrels(2025)
        if 'last_name, first_name' in raw_barrels.columns:
            raw_barrels['Name'] = raw_barrels['last_name, first_name'].apply(
                lambda x: ' '.join(reversed(x.split(', '))) if isinstance(x, str) else x)
        p_barrels = raw_barrels[['Name', 'brl_percent']].rename(
            columns={'brl_percent': 'Pitcher_Brl%'}) if not raw_barrels.empty else pd.DataFrame(
            columns=['Name', 'Pitcher_Brl%'])

        p26, p25 = pitching_stats(2026), pitching_stats(2025)
        p_merge = pd.merge(p26, p25, on='Name', how='outer', suffixes=('_26', '_25')).fillna(0)

        # Ensure columns exist before weight calc
        for col in ['SIERA_26', 'SIERA_25', 'K-BB%_26', 'K-BB%_25']:
            if col not in p_merge.columns: p_merge[col] = 0

        p_merge['W_SIERA'] = (p_merge['SIERA_26'] * 0.7) + (p_merge['SIERA_25'] * 0.3)
        p_merge['W_KBB'] = (p_merge['K-BB%_26'] * 0.7) + (p_merge['K-BB%_25'] * 0.3)
        p_combined = pd.merge(p_merge, p_barrels, on='Name', how='left').fillna(0)
        p_combined['Chalk_Quality'] = (p_combined['W_KBB'] * 120) + (15 / p_combined['W_SIERA'].replace(0, 5))
        p_combined['norm_name'] = p_combined['Name'].apply(normalize_name)
        p_final = p_combined[['norm_name', 'Chalk_Quality', 'W_SIERA', 'W_KBB', 'Pitcher_Brl%']].rename(
            columns={'W_SIERA': 'SIERA', 'W_KBB': 'K-BB%'})

        _STATS_CACHE['h'], _STATS_CACHE['p'], _STATS_CACHE['time'] = h_final, p_final, time.time()
        return h_final, p_final
    except:
        return pd.DataFrame(), pd.DataFrame()


def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4, excluded_games=[],
                  exposure_limit=1.0):
    df = df_input.copy()
    all_results, used_player_indices = [], []
    player_usage = {p: 0 for p in df.index}
    max_count = max(1, int(num_lineups * exposure_limit))

    if excluded_games:
        for gid in excluded_games:
            teams = gid.split(" vs ")
            df = df[~df['Team'].isin(teams)]

    p_hand_map = df[df['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()
    p_barrel_map = df[df['POS'].str.contains('P', na=False)].set_index('Team')['Pitcher_Brl%'].to_dict()

    def apply_logic(row):
        proj = row['Proj']
        if 'P' in str(row['POS']):
            proj *= (1.1 + (row.get('Chalk_Quality', 0) / 180))
        else:
            if row.get('Edge_Value', 0) > 0: proj = (proj * 0.5) + (row['Edge_Value'] * 0.5)
            if p_barrel_map.get(row['Opponent'], 0) > 10.5: proj *= 1.25
            b_h, o_h = row['CleanHand'], p_hand_map.get(row['Opponent'], '?')
            if b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L'): proj *= 1.12
        return proj * random.uniform(0.96, 1.04)

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    teams_to_stack = [stack_team] if stack_team and stack_team != "None" else \
    df[~df['POS'].str.contains('P')].groupby('Team')['Solver_Proj'].mean().sort_values(ascending=False).head(
        8).index.tolist()

    for i in range(num_lineups):
        best_lineup = None
        highest_score = -1
        for current_team in teams_to_stack:
            try:
                prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
                players, slots = df.index.tolist(), list(POS_ORDER.keys())
                x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")
                prob += pulp.lpSum([df.loc[p, 'Solver_Proj'] * x[p][s] for p in players for s in slots])
                prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000
                for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
                for p in players:
                    prob += pulp.lpSum([x[p][s] for s in slots]) <= 1
                    if player_usage[p] >= max_count:
                        prob += pulp.lpSum([x[p][s] for s in slots]) == 0
                    elif df.loc[p, 'Player'] in locks:
                        prob += pulp.lpSum([x[p][s] for s in slots]) == 1
                    pos = str(df.loc[p, 'POS'])
                    for s in slots:
                        if (s.startswith('P') and 'P' not in pos) or (s == 'C' and 'C' not in pos) or (
                                s == '1B' and '1B' not in pos) or (s == '2B' and '2B' not in pos) or (
                                s == '3B' and '3B' not in pos) or (s == 'SS' and 'SS' not in pos) or (
                                s.startswith('OF') and 'OF' not in pos): prob += x[p][s] == 0
                for past in used_player_indices: prob += pulp.lpSum([x[p][s] for p in past for s in slots]) <= (
                            len(slots) - diversity)
                h_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P'))].index.tolist()
                if len(h_idx) >= int(min_stack):
                    prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= int(min_stack)
                else:
                    continue
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=1))
                if pulp.LpStatus[prob.status] in ['Optimal', 'Feasible'] and pulp.value(prob.objective) > highest_score:
                    highest_score = pulp.value(prob.objective)
                    l_players, p_idx, t_sal, t_proj = [], [], 0, 0
                    for p in players:
                        for s in slots:
                            if x[p][s].varValue == 1:
                                row = df.loc[p]
                                l_players.append({'Slot': s, 'Name': row['Player'], 'Team': row['Team'],
                                                  'Logo': get_logo_url(row['Team']),
                                                  'Proj': round(row['Solver_Proj'], 2), 'Salary': row['Salary'],
                                                  'SortKey': POS_ORDER[s]})
                                p_idx.append(p);
                                t_sal += row['Salary'];
                                t_proj += row['Solver_Proj']
                    l_players.sort(key=lambda x: x['SortKey'])
                    best_lineup = {'players': l_players, 'total_salary': t_sal, 'total_projection': round(t_proj, 2),
                                   'indices': p_idx}
            except:
                continue
        if best_lineup:
            all_results.append(best_lineup);
            used_player_indices.append(best_lineup['indices'])
            for idx in best_lineup['indices']: player_usage[idx] += 1
        else:
            break
    return all_results


@app.route('/', methods=['GET', 'POST'])
def index():
    espn_times = get_espn_game_times()
    h_fg, p_fg = get_weighted_stats()
    df_raw = pd.read_csv(SALARY_CSV)
    confirmed_teams = set()
    order_col = next((c for c in df_raw.columns if 'order' in c.lower()), None)
    if order_col: confirmed_teams = set(
        df_raw[pd.to_numeric(df_raw[order_col], errors='coerce').between(1, 9)]['Team'].unique())

    df_raw['norm_name'] = df_raw['Player'].apply(normalize_name)
    df_raw['CleanHand'] = df_raw['Hand'].apply(clean_hand_str)
    df_raw['Salary'] = pd.to_numeric(df_raw['Salary'].astype(str).replace(r'[\$,]', '', regex=True).apply(
        lambda x: float(x.replace('k', '')) * 1000 if 'k' in str(x).lower() else x), errors='coerce').fillna(0)
    df_raw['Proj'] = pd.to_numeric(df_raw['Projected Points'], errors='coerce').fillna(0)
    if not h_fg.empty: df_raw = df_raw.merge(h_fg, on='norm_name', how='left').fillna(0)
    if not p_fg.empty: df_raw = df_raw.merge(p_fg, on='norm_name', how='left').fillna(0)

    p_hand_map = df_raw[df_raw['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()
    pool_list, unique_game_map = [], {}
    for _, r in df_raw.iterrows():
        p_data = r.to_dict()
        p_data['Logo'] = get_logo_url(r['Team'])
        o_h = p_hand_map.get(r['Opponent'], '?')
        p_data['OppP'] = "—" if 'P' in str(r['POS']) else o_h
        p_data['Adv'] = False if 'P' in str(r['POS']) else (
                    r['CleanHand'] == 'S' or (r['CleanHand'] == 'L' and o_h == 'R') or (
                        r['CleanHand'] == 'R' and o_h == 'L'))
        pool_list.append(p_data)
        g_id = " vs ".join(sorted(
            [TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))]))
        if g_id not in unique_game_map: unique_game_map[g_id] = {'t1': r['Team'], 't2': r['Opponent']}

    available_games = []
    for g_id, teams in unique_game_map.items():
        time_data = espn_times.get(g_id, {'display': 'TBD', 'raw': datetime.max.replace(tzinfo=pytz.utc)})
        available_games.append({
            "id": g_id, "display": g_id, "time": time_data['display'], "sort_time": time_data['raw'],
            "t1": teams['t1'], "t2": teams['t2'], "l1": get_logo_url(teams['t1']), "l2": get_logo_url(teams['t2']),
            "i1": '<span class="status-dot on"></span>' if teams[
                                                               't1'] in confirmed_teams else '<span class="status-dot off"></span>',
            "i2": '<span class="status-dot on"></span>' if teams[
                                                               't2'] in confirmed_teams else '<span class="status-dot off"></span>'
        })
    available_games.sort(key=lambda x: x['sort_time'])

    results, status = None, "SYSTEMS LIVE"
    if request.method == 'POST':
        results = run_optimizer(df_raw, num_lineups=int(request.form.get('num_lineups', 5)),
                                locks=request.form.getlist('player_locks'), stack_team=request.form.get('stack_team'),
                                min_stack=request.form.get('min_stack', 3),
                                diversity=int(request.form.get('diversity', 4)),
                                exposure_limit=float(request.form.get('exposure_limit', 1.0)),
                                excluded_games=[g['id'] for g in available_games if
                                                g['id'] not in request.form.getlist('games')])
        status = f"LOCKED {len(results)} LINEUPS"
    return render_template_string(HTML_BODY, results=results, status=status,
                                  teams=sorted(df_raw['Team'].dropna().unique()), games=available_games, pool=pool_list)


HTML_BODY = """
<!DOCTYPE html>
<html>
<head>
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-V4NJH4K19B"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-V4NJH4K19B');
    </script>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --accent: #3fb950; --text: #c9d1d9; --header: #161b22; --win: #238636; --loss: #da3633; --h-edge: #e3b341; --p-edge: #58a6ff; --row-alt: #1c2128; --logo-bg: rgba(255, 255, 255, 0.12); }
        body { background: var(--bg); color: var(--text); font-family: -apple-system, system-ui, sans-serif; padding: 10px; margin: 0; line-height: 1.5; }
        .container { max-width: 1400px; margin: auto; }
        .top-nav { background: var(--header); padding: 12px 15px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid var(--border); margin-bottom: 15px; position: sticky; top: 0; z-index: 100; }
        .status-pill { background: #000; border: 1px solid var(--accent); color: var(--accent); padding: 4px 10px; border-radius: 20px; font-weight: 800; font-size: 0.7em; letter-spacing: 0.5px; }
        .grid-layout { display: flex; flex-direction: column; gap: 15px; }
        @media (min-width: 992px) { .grid-layout { display: grid; grid-template-columns: 320px 1fr; gap: 20px; align-items: start; } body { padding: 20px; } }
        .sidebar { display: flex; flex-direction: column; gap: 15px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .card h3 { margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 10px; font-size: 0.85em; text-transform: uppercase; color: #f0f6fc; letter-spacing: 0.5px; }
        .table-responsive { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85em; white-space: nowrap; }
        th { text-align: left; background: var(--header); padding: 12px 10px; color: #8b949e; text-transform: uppercase; font-size: 0.65em; border-bottom: 2px solid var(--border); }
        tbody tr:nth-child(even) { background-color: var(--row-alt); }
        td { padding: 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
        .stat-val { font-family: monospace; font-size: 1em; }
        .player-name { color: #f0f6fc; font-weight: 600; white-space: normal; min-width: 120px; }
        .team-logo-icon { width: 24px; height: 24px; vertical-align: middle; margin-right: 8px; background: var(--logo-bg); border-radius: 4px; padding: 2px; }
        .slate-logo { width: 20px; height: 20px; vertical-align: middle; background: var(--logo-bg); border-radius: 4px; padding: 2px; margin: 0 4px; }
        .game-card { background: var(--row-alt); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 8px; }
        .status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin: 0 4px; }
        .status-dot.on { background: var(--accent); }
        .status-dot.off { background: #484f58; }
        button.lock-btn { background: var(--win); color: white; border: none; font-weight: 700; cursor: pointer; padding: 14px; border-radius: 6px; width: 100%; font-size: 0.9em; }
        input, select { background: #0d1117; color: #fff; border: 1px solid var(--border); padding: 10px; border-radius: 6px; font-size: 16px; }
        .hand-tag { padding: 2px 6px; border-radius: 3px; font-weight: 700; font-size: 0.7em; }
        .adv-match { background: var(--win); color: #fff; }
        .neut-match { background: #30363d; color: #8b949e; }
        .pool-scroll { max-height: 60vh; overflow-y: auto; }
        .lineup-grid { display: grid; grid-template-columns: 1fr; gap: 15px; margin-top: 25px; }
        @media (min-width: 768px) { .lineup-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (min-width: 1200px) { .lineup-grid { grid-template-columns: repeat(3, 1fr); } }
    </style>
</head>
<body>
<div class="container">
    <div class="top-nav">
        <h2 style="margin:0; font-size: 1.1em; color:#f0f6fc;">BETIFY <span style="color:var(--accent)">PRO</span></h2>
        <div class="status-pill">{{ status }}</div>
    </div>
    <form method="post">
    <div class="grid-layout">
        <div class="sidebar">
            <div class="card">
                <h3>Draft Config</h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:12px;">
                    <div><small style="font-size:0.6em; color:#8b949e">LINEUPS</small><input type="number" name="num_lineups" value="10" style="width:90%"></div>
                    <div><small style="font-size:0.6em; color:#8b949e">DIVERSITY</small><input type="number" name="diversity" value="4" style="width:90%"></div>
                    <div><small style="font-size:0.6em; color:#8b949e">MIN STACK</small><input type="number" name="min_stack" value="3" style="width:90%"></div>
                    <div><small style="font-size:0.6em; color:#8b949e">EXPOSURE</small><select name="exposure_limit" style="width:100%"><option value="1.0">100%</option><option value="0.5">50%</option></select></div>
                </div>
                <small style="font-size:0.6em; color:#8b949e">STACK TEAM</small>
                <select name="stack_team" style="margin-bottom:15px; width:100%"><option value="None">AUTO SELECT</option>{% for t in teams %}<option value="{{ t }}">{{ t }}</option>{% endfor %}</select>
                <button type="submit" class="lock-btn">LOCK LINEUPS</button>
            </div>
            <div class="card">
                <h3>Game Slate</h3>
                <div style="max-height:220px; overflow-y:auto;">
                    {% for g in games %}
                    <div class="game-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.7em; color:var(--accent); font-family:monospace;">{{ g.time }} ET</span>
                            <input type="checkbox" name="games" value="{{ g.id }}" checked style="width:18px; height:18px;">
                        </div>
                        <div style="display:flex; align-items:center; justify-content:space-between; font-size:0.9em;">
                            <span><img src="{{ g.l1 }}" class="slate-logo">{{ g.i1|safe }}<b style="color:#f0f6fc">{{ g.t1 }}</b></span>
                            <span style="color:var(--border);">vs</span>
                            <span><b style="color:#f0f6fc">{{ g.t2 }}</b>{{ g.i2|safe }}<img src="{{ g.l2 }}" class="slate-logo"></span>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        <div class="card">
            <div style="display:flex; flex-direction:column; gap:10px; margin-bottom:15px;">
                <h3 style="margin:0; border:none;">Player Pool</h3>
                <input type="text" id="pSearch" placeholder="Filter by name, team, or position..." style="width:100%; box-sizing:border-box;">
            </div>
            <div class="pool-scroll">
                <div class="table-responsive">
                    <table>
                        <thead><tr><th>LCK</th><th>PLAYER</th><th>POS</th><th>TEAM</th><th>BATS</th><th>%</th><th>RAW</th><th>PWR</th><th>SAL</th><th>PRJ</th></tr></thead>
                        <tbody id="pBody">
                            {% for p in pool %}
                            <tr>
                                <td><input type="checkbox" name="player_locks" value="{{ p.get('Player','') }}" style="width:18px; height:18px;"></td>
                                <td><img src="{{ p.get('Logo','') }}" class="team-logo-icon"><span class="player-name">{{ p.get('Player','') }}</span></td>
                                <td style="color:var(--accent); font-weight:bold;">{{ p.get('POS','') }}</td>
                                <td style="color:#8b949e">{{ p.get('Team','') }}</td>
                                <td><span class="hand-tag {{ 'adv-match' if p.get('Adv') else 'neut-match' }}">{{ p.get('Hand','') }}</span></td>
                                <td class="stat-val" style="color:{{ 'var(--p-edge)' if 'P' in p.get('POS','') else 'var(--h-edge)' }}">
                                    {{ (p.get('K-BB%', 0)*100)|round(0) if 'P' in p.get('POS','') else (p.get('Barrel%', 0)*100)|round(0) }}%
                                </td>
                                <td class="stat-val">
                                    {{ p.get('SIERA', 0)|round(2) if 'P' in p.get('POS','') else p.get('xwOBA', 0)|round(3) }}
                                </td>
                                <td class="stat-val" style="color:var(--accent)">
                                    {{ p.get('Chalk_Quality', 0)|round(0) if 'P' in p.get('POS','') else p.get('Edge_Value', 0)|round(0) }}
                                </td>
                                <td class="stat-val" style="color:#f0f6fc">${{ "{:,.0f}".format(p.get('Salary',0)) }}</td>
                                <td class="stat-val" style="color:#f0f6fc; font-weight:700;">{{ p.get('Proj',0)|round(1) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    </form>
    {% if results %}
    <div class="lineup-grid">
        {% for lineup in results %}
        <div class="card" style="border-top: 3px solid var(--accent); padding: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <b style="color:#f0f6fc; font-size:0.8em;">LINEUP #{{ loop.index }}</b>
                <span class="stat-val" style="color:var(--accent); font-size:0.9em;">{{ lineup.total_projection }} <small style="color:#8b949e">PTS</small></span>
            </div>
            <table style="font-size:0.8em;">
                {% for p in lineup.players %}
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                    <td style="color:var(--accent); font-weight:bold; width:25px;">{{p.Slot}}</td>
                    <td><img src="{{p.Logo}}" class="team-logo-icon" style="width:20px; height:20px;"><span class="player-name" style="font-size:0.9em;">{{p.Name}}</span></td>
                    <td class="stat-val" style="text-align:right; color:#8b949e">${{ "{:,.0f}".format(p.Salary) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
<script>
    document.getElementById('pSearch').addEventListener('input', function() {
        let q = this.value.toLowerCase();
        document.querySelectorAll('#pBody tr').forEach(r => {
            r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
        });
    });
</script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)