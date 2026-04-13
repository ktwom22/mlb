import pandas as pd
import pulp
import re
import unicodedata
import random
import time
import os
from flask import Flask, render_template_string, request
from datetime import datetime
import pytz
import requests
from pybaseball import pitching_stats, batting_stats, batting_stats_bref, pitching_stats_bref
from unidecode import unidecode

app = Flask(__name__)

# --- CACHE CONTROL ---
_STATS_CACHE = {'h': None, 'p': None, 'time': 0}

# --- CONFIG ---
SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"
POS_ORDER = {'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4, '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9}

TEAM_MAP = {
    "CHW": "CWS", "CHA": "CWS", "CWS": "CWS", "WSH": "WAS", "WAS": "WAS", "Washington": "WAS",
    "OAK": "OAK", "ATH": "OAK", "SF": "SF", "SFO": "SF", "SFG": "SF", "San Francisco": "SF",
    "AZ": "ARI", "ARI": "ARI", "Arizona": "ARI", "TB": "TB", "TBA": "TB", "Tampa Bay": "TB",
    "KC": "KC", "KCA": "KC", "Kansas City": "KC", "SD": "SD", "SDN": "SD", "San Diego": "SD",
    "NYY": "NYY", "NYA": "NYY", "New York": "NYM", "NYM": "NYM", "NYN": "NYM",
    "LAD": "LAD", "LAN": "LAD", "Los Angeles": "LAD", "STL": "STL", "SLN": "STL", "St. Louis": "STL",
    "CHC": "CHC", "CHN": "CHC", "Chicago": "CHC", "TOR": "TOR", "Toronto": "TOR",
    "COL": "COL", "Colorado": "COL", "ATL": "ATL", "Atlanta": "ATL", "Boston": "BOS",
    "Miami": "MIA", "Philadelphia": "PHI", "Cleveland": "CLE", "Detroit": "DET",
    "Houston": "HOU", "Milwaukee": "MIL", "Minnesota": "MIN", "Pittsburgh": "PIT",
    "Seattle": "SEA", "Texas": "TEX", "Baltimore": "BAL", "Cincinnati": "CIN", "Anaheim": "LAA"
}

TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
    "SEA": 136, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WAS": 120
}

# --- HELPERS ---

def get_logo_url(team_abbr):
    clean_abbr = TEAM_MAP.get(team_abbr, team_abbr)
    tid = TEAM_ID_MAP.get(clean_abbr)
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{tid}.svg" if tid else "https://www.mlbstatic.com/team-logos/league/1.svg"

def normalize_name(name):
    if not isinstance(name, str): return ""
    junk_map = {
        r'\\xc3\\xb1': 'n', r'\\xc3\\xa1': 'a', r'\\xc3\\xa9': 'e',
        r'\\xc3\\xad': 'i', r'\\xc3\\xb3': 'o', r'\\xc3\\xba': 'u', r'\\xc3\\xbc': 'u'
    }
    for pattern, rep in junk_map.items():
        name = re.sub(pattern, rep, name)
    name = unidecode(name).lower()
    name = re.sub(r'[^a-z\s]', '', name).strip()
    for s in [' jr', ' sr', ' iii', ' ii', ' iv']:
        if name.endswith(s): name = name[:-len(s)]
    return name.strip()

def clean_hand_str(h):
    if pd.isna(h): return "?"
    s = str(h).upper()
    return s[0] if s and s[0] in 'RLS' else "?"

def get_mlb_weather_data():
    weather_map = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        sched_url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date={today}"
        sched_data = requests.get(sched_url, headers=headers, timeout=5).json()
        for date_info in sched_data.get('dates', []):
            for game in date_info.get('games', []):
                try:
                    away = TEAM_MAP.get(game['teams']['away']['team']['abbreviation'], game['teams']['away']['team']['abbreviation'])
                    home = TEAM_MAP.get(game['teams']['home']['team']['abbreviation'], game['teams']['home']['team']['abbreviation'])
                    game_id = " vs ".join(sorted([away, home]))
                    pk = game['gamePk']
                    live_url = f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
                    live_data = requests.get(live_url, headers=headers, timeout=3).json()
                    w = live_data.get('gameData', {}).get('weather', {})
                    weather_map[game_id] = {
                        'temp': int(w.get('temp', 70)) if str(w.get('temp')).isdigit() else 70,
                        'wind': w.get('wind', '0 mph, Calm'),
                        'condition': w.get('condition', 'Unknown')
                    }
                except: continue
    except: pass
    return weather_map

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
            raw_names = [TEAM_MAP.get(t['team']['abbreviation'].upper(), t['team']['abbreviation'].upper()) for t in competitors]
            game_id = " vs ".join(sorted(raw_names))
            utc_time = datetime.strptime(event['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=pytz.utc)
            et_dt = utc_time.astimezone(pytz.timezone('US/Eastern'))
            slate_times[game_id] = {'display': et_dt.strftime('%I:%M %p'), 'raw': et_dt}
    except: pass
    return slate_times

def get_weighted_stats():
    global _STATS_CACHE
    if _STATS_CACHE['h'] is not None and (time.time() - _STATS_CACHE['time']) < 3600:
        return _STATS_CACHE['h'], _STATS_CACHE['p']
    try:
        try:
            h_df = batting_stats(2025, qual=50)
            use_fg = True
        except:
            h_df = batting_stats_bref(2025)
            use_fg = False
        h_df['norm_name'] = h_df['Name'].apply(normalize_name)
        if use_fg:
            h_df['Edge_Value'] = (h_df['Barrel%'] * 50) + (h_df['xwOBA'] * 20)
        else:
            h_df['Edge_Value'] = (pd.to_numeric(h_df['OPS'], errors='coerce').fillna(0) * 100) + (pd.to_numeric(h_df['HR'], errors='coerce').fillna(0) * 2)
        h_final = h_df[['norm_name', 'Edge_Value']]
        try:
            p_df = pitching_stats(2025, qual=20)
            use_fg_p = True
        except:
            p_df = pitching_stats_bref(2025)
            use_fg_p = False
        p_df['norm_name'] = p_df['Name'].apply(normalize_name)
        if use_fg_p:
            p_df['Chalk_Quality'] = (p_df['K-BB%'] * 120) + (15 / p_df['SIERA'].replace(0, 5))
        else:
            p_df['Chalk_Quality'] = (pd.to_numeric(p_df['SO'], errors='coerce').fillna(0) * 0.5) + (20 / pd.to_numeric(p_df['ERA'], errors='coerce').replace(0, 5))
        p_final = p_df[['norm_name', 'Chalk_Quality']]
        _STATS_CACHE['h'], _STATS_CACHE['p'], _STATS_CACHE['time'] = h_final, p_final, time.time()
    except Exception as e:
        print(f"Stats Error: {e}")
        h_final = pd.DataFrame(columns=['norm_name', 'Edge_Value'])
        p_final = pd.DataFrame(columns=['norm_name', 'Chalk_Quality'])
    return h_final, p_final

# --- OPTIMIZER ---

def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4, excluded_games=[],
                  exposure_limit=1.0, weather_data={}):
    df = df_input.copy()
    all_results, used_player_indices = [], []
    player_usage = {p: 0 for p in df.index}
    max_count = max(1, int(num_lineups * exposure_limit))
    p_hand_map = df[df['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()

    def apply_logic(row):
        proj = float(row.get('Proj', 5.0))
        game_id = " vs ".join(sorted([TEAM_MAP.get(row['Team'], row['Team']), TEAM_MAP.get(row['Opponent'], row['Opponent'])]))
        weather = weather_data.get(game_id, {'temp': 70, 'wind': '0 mph, Calm'})
        if 'P' in str(row['POS']):
            proj *= (1.1 + (float(row.get('Chalk_Quality', 0)) / 180))
        else:
            order = row.get('Order', 0)
            if order == 0: proj *= 0.10
            elif order <= 2: proj *= 1.25
            elif order <= 5: proj *= 1.15
            if float(row.get('Edge_Value', 0)) > 0:
                proj = (proj * 0.4) + (row['Edge_Value'] * 0.1)
            b_h, o_h = row['CleanHand'], p_hand_map.get(row['Opponent'], '?')
            if b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L'):
                proj *= 1.12
            if weather['temp'] >= 85: proj *= 1.05
            if 'out' in str(weather['wind']).lower(): proj *= 1.08
        return proj * random.uniform(0.95, 1.05)

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(r['Team'], r['Team']), TEAM_MAP.get(r['Opponent'], r['Opponent'])])) in excluded_games, axis=1)]

    teams_to_stack = [stack_team] if stack_team and stack_team != "None" else \
        df[~df['POS'].str.contains('P')].groupby('Team')['Solver_Proj'].mean().sort_values(ascending=False).head(8).index.tolist()

    for i in range(num_lineups):
        best_lineup, highest_score = None, -1
        random.shuffle(teams_to_stack)
        for current_team in teams_to_stack[:4]:
            try:
                prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
                players = df.index.tolist()
                random.shuffle(players)
                slots = list(POS_ORDER.keys())
                x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")
                prob += pulp.lpSum([df.loc[p, 'Solver_Proj'] * x[p][s] for p in players for s in slots])
                prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000
                for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
                for p in players:
                    prob += pulp.lpSum([x[p][s] for s in slots]) <= 1
                    if player_usage[p] >= max_count: prob += pulp.lpSum([x[p][s] for s in slots]) == 0
                    if df.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in slots]) == 1
                    pos = str(df.loc[p, 'POS'])
                    for s in slots:
                        if (s.startswith('P') and 'P' not in pos) or (s == 'C' and 'C' not in pos) or \
                                (s == '1B' and '1B' not in pos) or (s == '2B' and '2B' not in pos) or \
                                (s == '3B' and '3B' not in pos) or (s == 'SS' and 'SS' not in pos) or \
                                (s.startswith('OF') and 'OF' not in pos): prob += x[p][s] == 0
                for past in used_player_indices:
                    prob += pulp.lpSum([x[p][s] for p in past for s in slots]) <= (len(slots) - diversity)
                h_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P'))].index.tolist()
                if len(h_idx) >= int(min_stack):
                    prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= int(min_stack)
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=1))
                if pulp.LpStatus[prob.status] == 'Optimal':
                    score = pulp.value(prob.objective)
                    if score > highest_score:
                        highest_score = score
                        l_players, p_idx, t_sal, t_proj = [], [], 0, 0
                        for p in players:
                            for s in slots:
                                if x[p][s].varValue == 1:
                                    row = df.loc[p]
                                    l_players.append({'Slot': s, 'Name': row['Player'], 'Team': row['Team'],
                                                      'Logo': get_logo_url(row['Team']),
                                                      'Proj': round(row['Solver_Proj'], 2), 'Salary': row['Salary'],
                                                      'SortKey': POS_ORDER[s]})
                                    p_idx.append(p)
                                    t_sal += row['Salary']
                                    t_proj += row['Solver_Proj']
                        l_players.sort(key=lambda x: x['SortKey'])
                        best_lineup = {'players': l_players, 'total_salary': t_sal, 'total_projection': round(t_proj, 2), 'indices': p_idx}
            except: continue
        if best_lineup:
            all_results.append(best_lineup)
            used_player_indices.append(best_lineup['indices'])
            for idx in best_lineup['indices']: player_usage[idx] += 1
        else: break
    return all_results

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_info = get_mlb_weather_data()
    espn_times = get_espn_game_times()
    h_fg, p_fg = get_weighted_stats()
    df_raw = pd.read_csv(SALARY_CSV)

    sal_col = next((c for c in ['Salary', 'salary', 'Sal'] if c in df_raw.columns), 'Salary')
    df_raw['Salary'] = pd.to_numeric(df_raw[sal_col].astype(str).replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    proj_col = next((c for c in ['Proj', 'Projected Points', 'FPTS', 'Projected', 'Points'] if c in df_raw.columns), None)
    df_raw['Proj'] = pd.to_numeric(df_raw[proj_col], errors='coerce').fillna(0) if proj_col else 0.0
    ord_col = next((c for c in ['Order', 'Batting Order', 'Lineup'] if c in df_raw.columns), 'Order')
    df_raw['Order'] = pd.to_numeric(df_raw[ord_col], errors='coerce').fillna(0) if ord_col in df_raw.columns else 0
    df_raw['norm_name'] = df_raw['Player'].apply(normalize_name)
    df_raw['CleanHand'] = df_raw['Hand'].apply(clean_hand_str) if 'Hand' in df_raw.columns else '?'

    if not h_fg.empty: df_raw = df_raw.merge(h_fg, on='norm_name', how='left')
    if not p_fg.empty: df_raw = df_raw.merge(p_fg, on='norm_name', how='left')
    for col in ['Edge_Value', 'Chalk_Quality']:
        if col not in df_raw.columns: df_raw[col] = 0.0
    df_raw = df_raw.fillna(0.0)

    confirmed_teams = set(df_raw[df_raw['Order'] > 0]['Team'].unique())
    p_hand_map = df_raw[df_raw['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()

    pool_list = []
    unique_game_map = {}
    for _, r in df_raw.iterrows():
        p_data = r.to_dict()
        p_data['Logo'] = get_logo_url(r['Team'])
        g_id = " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))]))
        w_cur = weather_info.get(g_id, {'temp': '--', 'wind': 'Calm'})
        p_data['Weather_Short'] = f"{w_cur['temp']}°"
        w_icon = "⚪"
        wind_str = str(w_cur['wind']).lower()
        if 'out' in wind_str: w_icon = "🔥"
        elif 'in' in wind_str: w_icon = "❄️"
        elif w_cur['temp'] != '--' and int(w_cur['temp']) > 85: w_icon = "☀️"
        p_data['W_Icon'] = w_icon
        o_h = p_hand_map.get(r['Opponent'], '?')
        p_data['Adv'] = False if 'P' in str(r['POS']) else (r['CleanHand'] == 'S' or (r['CleanHand'] == 'L' and o_h == 'R') or (r['CleanHand'] == 'R' and o_h == 'L'))
        pool_list.append(p_data)
        if g_id not in unique_game_map: unique_game_map[g_id] = {'t1': r['Team'], 't2': r['Opponent']}

    available_games = []
    for g_id, teams in unique_game_map.items():
        time_data = espn_times.get(g_id, {'display': 'TBD', 'raw': datetime.max.replace(tzinfo=pytz.utc)})
        w_cur = weather_info.get(g_id, {'temp': '--', 'wind': 'Calm'})
        available_games.append({
            "id": g_id, "display": g_id, "time": time_data['display'], "sort_time": time_data['raw'],
            "t1": teams['t1'], "t2": teams['t2'], "l1": get_logo_url(teams['t1']), "l2": get_logo_url(teams['t2']),
            "i1": '<span class="status-dot on"></span>' if teams['t1'] in confirmed_teams else '<span class="status-dot off"></span>',
            "i2": '<span class="status-dot on"></span>' if teams['t2'] in confirmed_teams else '<span class="status-dot off"></span>',
            "weather": f"{w_cur['temp']}° {w_cur['wind']}"
        })
    available_games.sort(key=lambda x: x['sort_time'])

    results, status = None, "SYSTEMS LIVE"
    if request.method == 'POST':
        results = run_optimizer(
            df_raw,
            num_lineups=int(request.form.get('num_lineups', 10)),
            locks=request.form.getlist('player_locks'),
            stack_team=request.form.get('stack_team'),
            min_stack=int(request.form.get('min_stack', 3)),
            diversity=int(request.form.get('diversity', 4)),
            exposure_limit=float(request.form.get('exposure_limit', 1.0)),
            excluded_games=[g['id'] for g in available_games if g['id'] not in request.form.getlist('games')],
            weather_data=weather_info
        )
        status = f"LOCKED {len(results)} LINEUPS"

    return render_template_string(HTML_BODY, results=results, status=status,
                                  teams=sorted(df_raw['Team'].dropna().unique()), games=available_games, pool=pool_list)

# --- HTML ---
HTML_BODY = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root { 
            --bg: #0d1117; --card: #161b22; --border: #30363d; --accent: #3fb950; 
            --text: #c9d1d9; --header: #161b22; --win: #238636; --row-alt: #1c2128; --logo-bg: rgba(255, 255, 255, 0.12); 
        }
        body { background: var(--bg); color: var(--text); font-family: -apple-system, system-ui, sans-serif; padding: 10px; margin: 0; }
        .container { max-width: 1400px; margin: auto; padding: 10px; }
        .top-nav { background: var(--header); padding: 12px 15px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid var(--border); margin-bottom: 15px; }
        .status-pill { border: 1px solid var(--accent); color: var(--accent); padding: 4px 10px; border-radius: 20px; font-weight: 800; font-size: 0.7em; }
        .grid-layout { display: flex; flex-direction: column; gap: 15px; }
        @media (min-width: 992px) { .grid-layout { display: grid; grid-template-columns: 320px 1fr; gap: 20px; align-items: start; } }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .pool-container { max-height: 75vh; overflow-y: auto; border: 1px solid var(--border); }
        table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
        thead th { position: sticky; top: 0; z-index: 10; background: var(--header); padding: 10px; text-align: left; color: #8b949e; }
        td { padding: 10px; border-bottom: 1px solid var(--border); }
        .stat-val { font-family: monospace; }
        .team-logo-icon { width: 22px; height: 22px; vertical-align: middle; margin-right: 8px; background: var(--logo-bg); border-radius: 4px; padding: 2px; }
        .status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin: 0 4px; }
        .status-dot.on { background: var(--accent); }
        .status-dot.off { background: #484f58; }
        .lock-btn { background: var(--win); color: white; border: none; font-weight: 700; cursor: pointer; padding: 14px; border-radius: 6px; width: 100%; font-size: 1.1em; }
        input, select { background: #0d1117; color: #fff; border: 1px solid var(--border); padding: 8px; border-radius: 6px; width: 100%; box-sizing: border-box; }
        .hand-tag { padding: 2px 6px; border-radius: 3px; font-weight: 700; font-size: 0.7em; }
        .adv-match { background: var(--win); color: #fff; }
        .neut-match { background: #30363d; color: #8b949e; }
        .lineup-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 20px; }
        .label-group { margin-bottom: 12px; }
        small { color: #8b949e; font-weight: bold; display: block; margin-bottom: 4px; }
    </style>
</head>
<body>
<div class="container">
    <div class="top-nav">
        <h2 style="margin:0;">BETIFY <span style="color:var(--accent)">PRO</span></h2>
        <div class="status-pill">{{ status }}</div>
    </div>
    <form method="post">
    <div class="grid-layout">
        <div class="sidebar">
            <div class="card">
                <h3>Draft Config</h3>
                <div class="label-group">
                    <small>STACK TEAM</small>
                    <select name="stack_team">
                        <option value="None">Auto-Optimized</option>
                        {% for team in teams %}
                        <option value="{{ team }}">{{ team }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div class="label-group"><small>LINEUPS</small><input type="number" name="num_lineups" value="10"></div>
                    <div class="label-group"><small>STACK SIZE</small><input type="number" name="min_stack" value="3"></div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div class="label-group"><small>DIVERSITY</small><input type="number" name="diversity" value="4"></div>
                    <div class="label-group"><small>EXPOSURE</small><input type="number" step="0.1" name="exposure_limit" value="1.0"></div>
                </div>
                <button type="submit" class="lock-btn">LOCK LINEUPS</button>
            </div>
            <div class="card">
                <h3>Games</h3>
                {% for g in games %}
                <div style="background:var(--row-alt); padding:10px; border-radius:6px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.75em; color:var(--accent);">
                        <span>{{ g.time }}</span>
                        <input type="checkbox" name="games" value="{{ g.id }}" checked>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span>{{ g.i1|safe }} {{ g.t1 }}</span>
                        <span style="color:var(--border)">vs</span>
                        <span>{{ g.t2 }} {{ g.i2|safe }}</span>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="card" style="padding: 0;">
            <div class="pool-container">
                <table>
                    <thead>
                        <tr><th>LCK</th><th>PLAYER</th><th>POS</th><th>ORD</th><th>TEAM</th><th>WTH</th><th>BATS</th><th>PWR</th><th>SAL</th><th>PRJ</th></tr>
                    </thead>
                    <tbody>
                        {% for p in pool %}
                        <tr>
                            <td><input type="checkbox" name="player_locks" value="{{ p.Player }}"></td>
                            <td><img src="{{ p.Logo }}" class="team-logo-icon"><b>{{ p.Player }}</b></td>
                            <td style="color:var(--accent); font-weight:bold;">{{ p.POS }}</td>
                            <td>{{ p.Order if p.Order > 0 else '—' }}</td>
                            <td>{{ p.Team }}</td>
                            <td>{{ p.W_Icon }} <small>{{ p.Weather_Short }}</small></td>
                            <td><span class="hand-tag {{ 'adv-match' if p.Adv else 'neut-match' }}">{{ p.Hand }}</span></td>
                            <td class="stat-val" style="color:var(--accent)">{{ p.Chalk_Quality|round(0) if 'P' in p.POS else p.Edge_Value|round(0) }}</td>
                            <td class="stat-val">${{ "{:,.0f}".format(p.Salary) }}</td>
                            <td class="stat-val" style="font-weight:bold;">{{ p.Proj|round(1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    </form>
    {% if results %}
    <div class="lineup-grid">
        {% for lineup in results %}
        <div class="card" style="border-top: 3px solid var(--accent);">
            <div style="display:flex; justify-content:space-between;">
                <b>LINEUP #{{ loop.index }}</b>
                <span style="color:var(--accent)">{{ lineup.total_projection }} PTS</span>
            </div>
            <table style="margin-top:10px;">
                {% for p in lineup.players %}
                <tr><td style="color:var(--accent); font-weight:bold;">{{ p.Slot }}</td><td>{{ p.Name }} ({{ p.Team }})</td><td style="text-align:right;">${{ "{:,.0f}".format(p.Salary) }}</td></tr>
                {% endfor %}
            </table>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)