import pandas as pd
import pulp
import re
import unicodedata
import random
import time
import os
import statsapi
from flask import Flask, render_template_string, request, Response
from datetime import datetime
import pytz
import requests
from unidecode import unidecode
from thefuzz import process, fuzz

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
    "Seattle": "SEA", "Texas": "TEX", "Baltimore": "BAL", "Cincinnati": "CIN", "LAA": "LAA", "Anaheim": "LAA"
}

TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
    "SEA": 136, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WAS": 120
}


# --- HELPERS ---

def clean_float(val, default=0.0):
    if val is None or str(val).strip() in ['-.--', '---', '-', '']:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_logo_url(team_abbr):
    clean_abbr = TEAM_MAP.get(team_abbr, team_abbr)
    tid = TEAM_ID_MAP.get(clean_abbr)
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{tid}.svg" if tid else "https://www.mlbstatic.com/team-logos/league/1.svg"


def normalize_name(name):
    if not isinstance(name, str): return ""
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name, flags=re.IGNORECASE)
    return "".join(filter(str.isalnum, name)).lower()


def clean_hand_str(h):
    if pd.isna(h): return "?"
    s = str(h).upper()
    return s[0] if s and s[0] in 'RLS' else "?"


# --- API STATS FETCHING ---

def fetch_pitcher_metrics(year):
    try:
        params = {'stats': 'season', 'season': year, 'group': 'pitching', 'playerPool': 'all', 'limit': 1500}
        data = statsapi.get('stats', params)
        if not data or 'stats' not in data or not data['stats']: return {}
        year_data = {}
        splits = data['stats'][0].get('splits', [])
        for s in splits:
            st = s.get('stat', {})
            player_name = s.get('player', {}).get('fullName', '')
            if not player_name: continue
            name_key = normalize_name(player_name)
            year_data[name_key] = {
                'full_name': player_name,
                'WHIP': clean_float(st.get('whip'), 1.35),
                'K9': clean_float(st.get('strikeoutsPer9Inn'), 7.5),
                'HR9': clean_float(st.get('homeRunsPer9'), 1.2),
                'K_BB': clean_float(st.get('strikeoutWalkRatio'), 2.5),
                'ERA': clean_float(st.get('era'), 4.50),
                'GS': int(st.get('gamesStarted', 0))
            }
        return year_data
    except Exception:
        return {}


def fetch_hitter_metrics(year):
    try:
        params = {'stats': 'season', 'season': year, 'group': 'hitting', 'playerPool': 'all', 'limit': 1500}
        data = statsapi.get('stats', params)
        if not data or 'stats' not in data or not data['stats']: return {}
        year_data = {}
        splits = data['stats'][0].get('splits', [])
        for s in splits:
            st = s.get('stat', {})
            player_name = s.get('player', {}).get('fullName', '')
            if not player_name: continue
            name_key = normalize_name(player_name)
            ops = clean_float(st.get('ops'), 0.700)
            year_data[name_key] = {
                'full_name': player_name,
                'OPS': ops,
                'ISO': clean_float(st.get('slg'), 0.400) - clean_float(st.get('avg'), 0.250),
                'wRC_proxy': int((ops / 0.750) * 100)
            }
        return year_data
    except Exception:
        return {}


def get_weighted_stats():
    global _STATS_CACHE
    if _STATS_CACHE['h'] is not None and (time.time() - _STATS_CACHE['time']) < 3600:
        return _STATS_CACHE['h'], _STATS_CACHE['p']
    W25, W26 = 0.7, 0.3
    p25, p26 = fetch_pitcher_metrics(2025), fetch_pitcher_metrics(2026)
    blended_p = []
    all_p_names = set(p25.keys()) | set(p26.keys())
    for name in all_p_names:
        s25, s26 = p25.get(name), p26.get(name)

        def blend_p(key, d):
            v25 = s25[key] if s25 else None
            v26 = s26[key] if s26 else None
            if v25 is not None and v26 is not None: return (v25 * W25) + (v26 * W26)
            return v26 if v26 is not None else (v25 if v25 is not None else d)

        k9, kbb, whip = blend_p('K9', 7.5), blend_p('K_BB', 2.5), blend_p('WHIP', 1.35)
        chalk = (k9 * 0.5) + (kbb * 0.5) - (whip * 2.0)
        blended_p.append({
            'norm_name': name, 'full_name': s26['full_name'] if s26 else s25['full_name'],
            'Chalk_Quality': round(max(0.1, chalk), 2), 'WHIP': round(whip, 2),
            'K/9': round(k9, 2), 'HR/9': blend_p('HR9', 1.2), 'GS': s26['GS'] if s26 else (s25['GS'] if s25 else 0)
        })
    h25, h26 = fetch_hitter_metrics(2025), fetch_hitter_metrics(2026)
    blended_h = []
    all_h_names = set(h25.keys()) | set(h26.keys())
    for name in all_h_names:
        s25, s26 = h25.get(name), h26.get(name)

        def blend_h(key, d):
            v25 = s25[key] if s25 else None
            v26 = s26[key] if s26 else None
            if v25 is not None and v26 is not None: return (v25 * W25) + (v26 * W26)
            return v26 if v26 is not None else (v25 if v25 is not None else d)

        iso = blend_h('ISO', 0.150)
        blended_h.append({
            'norm_name': name, 'full_name': s26['full_name'] if s26 else s25['full_name'],
            'ISO': round(iso, 3), 'wRC+': int(blend_h('wRC_proxy', 100)), 'Edge_Value': round(iso * 400, 2)
        })
    h_df, p_df = pd.DataFrame(blended_h), pd.DataFrame(blended_p)
    _STATS_CACHE.update({'h': h_df, 'p': p_df, 'time': time.time()})
    return h_df, p_df


# --- OPTIMIZER ---

def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4, excluded_games=[],
                  exposure_limit=1.0, weather_data={}):
    df = df_input.copy()
    if excluded_games:
        df = df[~df.apply(lambda r: " vs ".join(sorted([TEAM_MAP.get(str(r['Team']), str(r['Team'])),
                                                        TEAM_MAP.get(str(r['Opponent']),
                                                                     str(r['Opponent']))])) in excluded_games, axis=1)]
    all_results, used_player_indices = [], []
    player_usage = {p: 0 for p in df.index}
    max_count = max(1, int(num_lineups * exposure_limit))
    p_hand_map = df[df['POS'].str.contains('P', na=False)].set_index('Team')['CleanHand'].to_dict()

    def apply_logic(row):
        proj_val = row.get('Kris Bubic projected points', row.get('Proj', 5.0))
        proj = float(proj_val) if float(proj_val) > 0 else 5.0
        t1, t2 = TEAM_MAP.get(str(row['Team']), str(row['Team'])), TEAM_MAP.get(str(row['Opponent']),
                                                                                str(row['Opponent']))
        game_id = " vs ".join(sorted([t1, t2]))
        w = weather_data.get(game_id, {'temp': 70, 'wind': '0 mph, Calm', 'condition': 'Clear'})
        if 'P' in str(row['POS']):
            proj += (float(row.get('Chalk_Quality', 0)) * 0.8)
        else:
            if row.get('Order', 0) == 0:
                proj *= 0.1
            elif row['Order'] <= 2:
                proj *= 1.25
            elif row['Order'] <= 5:
                proj *= 1.15
            if float(row.get('Edge_Value', 0)) > 0: proj = (proj * 0.5) + (row['Edge_Value'] * 0.15)
            b_h, o_h = row.get('CleanHand', '?'), p_hand_map.get(row['Opponent'], '?')
            if b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L'): proj *= 1.15
            wind, cond = str(w['wind']).lower(), str(w['condition']).lower()
            if "dome" not in cond:
                if isinstance(w['temp'], int) and w['temp'] >= 85: proj *= 1.10
                if 'out' in wind:
                    proj *= 1.08
                elif 'in' in wind:
                    proj *= 0.92
        return proj * random.uniform(0.97, 1.03)

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    teams_to_stack = [stack_team] if stack_team and stack_team != "None" else \
        df[~df['POS'].str.contains('P')].groupby('Team')['Solver_Proj'].mean().sort_values(ascending=False).head(
            8).index.tolist()

    for i in range(num_lineups):
        best_lineup = None
        highest_score = -1
        random.shuffle(teams_to_stack)
        for current_team in teams_to_stack[:3]:
            try:
                prob = pulp.LpProblem(f"MLB_{i}_{current_team}", pulp.LpMaximize)
                players = df.index.tolist()
                slots = list(POS_ORDER.keys())
                x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")
                prob += pulp.lpSum([df.loc[p, 'Solver_Proj'] * x[p][s] for p in players for s in slots])
                prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000
                for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
                for p in players:
                    prob += pulp.lpSum([x[p][s] for s in slots]) <= 1
                    if df.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in slots]) == 1
                    if player_usage.get(p, 0) >= max_count: prob += pulp.lpSum([x[p][s] for s in slots]) == 0
                    pos = str(df.loc[p, 'POS'])
                    for s in slots:
                        valid = any(
                            [(s.startswith('P') and 'P' in pos), (s == 'C' and 'C' in pos), (s == '1B' and '1B' in pos),
                             (s == '2B' and '2B' in pos), (s == '3B' and '3B' in pos), (s == 'SS' and 'SS' in pos),
                             (s.startswith('OF') and 'OF' in pos)])
                        if not valid: prob += x[p][s] == 0
                for past in used_player_indices: prob += pulp.lpSum([x[p][s] for p in past for s in slots]) <= (
                            len(slots) - diversity)
                h_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P'))].index.tolist()
                if len(h_idx) >= int(min_stack): prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= int(
                    min_stack)
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=3))
                if pulp.LpStatus[prob.status] == 'Optimal':
                    score = pulp.value(prob.objective)
                    if score > highest_score:
                        highest_score = score
                        lineup_data, p_indices, t_sal, t_proj = [], [], 0, 0
                        for p in players:
                            for s in slots:
                                if pulp.value(x[p][s]) == 1:
                                    row = df.loc[p]
                                    lineup_data.append({'Slot': s, 'Name': row['Player'], 'Team': row['Team'],
                                                        'Logo': get_logo_url(row['Team']),
                                                        'Proj': round(row['Solver_Proj'], 2), 'Salary': row['Salary'],
                                                        'SortKey': POS_ORDER[s]})
                                    p_indices.append(p);
                                    t_sal += row['Salary'];
                                    t_proj += row['Solver_Proj']
                        lineup_data.sort(key=lambda x: x['SortKey'])
                        best_lineup = {'players': lineup_data, 'total_salary': t_sal,
                                       'total_projection': round(t_proj, 2), 'indices': p_indices}
            except:
                continue
        if best_lineup:
            all_results.append(best_lineup)
            used_player_indices.append(best_lineup['indices'])
            for idx in best_lineup['indices']: player_usage[idx] += 1
        else:
            break
    return all_results


# --- SEO & UTILITY ROUTES ---

@app.route('/robots.txt')
def robots():
    lines = ["User-agent: *", "Disallow: /static/", "Allow: /", f"Sitemap: {request.url_root.rstrip('/')}/sitemap.xml"]
    return Response("\n".join(lines), mimetype="text/plain")


@app.route('/sitemap.xml')
def sitemap():
    pages = [[f"{request.url_root.rstrip('/')}/", "daily"]]
    xml_sitemap = render_template_string('''<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            {% for page in pages %}
            <url><loc>{{ page[0] }}</loc><changefreq>{{ page[1] }}</changefreq><priority>1.0</priority></url>
            {% endfor %}
        </urlset>''', pages=pages)
    return Response(xml_sitemap, mimetype="application/xml")


# --- MAIN ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_info = get_mlb_weather_data()
    espn_times = get_espn_game_times()
    h_fg, p_fg = get_weighted_stats()
    df_raw = pd.read_csv(SALARY_CSV)
    sal_col = next((c for c in ['Salary', 'salary', 'Sal'] if c in df_raw.columns), 'Salary')
    proj_col = next((c for c in ['Kris Bubic projected points', 'Proj', 'FPTS', 'Points'] if c in df_raw.columns), None)
    df_raw['Salary'] = pd.to_numeric(df_raw[sal_col].astype(str).replace(r'[\$,]', '', regex=True),
                                     errors='coerce').fillna(0)
    df_raw['Proj_Base'] = pd.to_numeric(df_raw[proj_col], errors='coerce').fillna(0.0) if proj_col else 0.0
    df_raw['Order'] = pd.to_numeric(df_raw['Order'], errors='coerce').fillna(0) if 'Order' in df_raw.columns else 0
    df_raw['CleanHand'] = df_raw['Hand'].apply(clean_hand_str) if 'Hand' in df_raw.columns else '?'
    for col in ['Edge_Value', 'Chalk_Quality', 'ISO', 'wRC+', 'WHIP', 'HR/9', 'K/9', 'GS']: df_raw[col] = 0.0
    h_choices = h_fg['full_name'].tolist()
    for idx, row in df_raw[~df_raw['POS'].str.contains('P')].iterrows():
        m = process.extractOne(row['Player'], h_choices, scorer=fuzz.token_set_ratio)
        if m and m[1] >= 75:
            s_row = h_fg[h_fg['full_name'] == m[0]].iloc[0]
            for c in ['Edge_Value', 'ISO', 'wRC+']: df_raw.at[idx, c] = s_row[c]
    p_choices = p_fg['full_name'].tolist()
    for idx, row in df_raw[df_raw['POS'].str.contains('P')].iterrows():
        m = process.extractOne(row['Player'], p_choices, scorer=fuzz.token_set_ratio)
        if m and m[1] >= 75:
            s_row = p_fg[p_fg['full_name'] == m[0]].iloc[0]
            for c in ['Chalk_Quality', 'WHIP', 'HR/9', 'K/9', 'GS']: df_raw.at[idx, c] = s_row[c]
    p_hand_map = df_raw[df_raw['POS'].str.contains('P')].set_index('Team')['CleanHand'].to_dict()

    def get_ui_proj(row):
        base = float(row['Proj_Base']) if float(row['Proj_Base']) > 0 else 5.0
        if 'P' in str(row['POS']):
            base += (float(row['Chalk_Quality']) * 0.8)
        else:
            if 0 < row['Order'] <= 2: base *= 1.25
            if row['Edge_Value'] > 0: base = (base * 0.5) + (row['Edge_Value'] * 0.15)
        return round(base, 2)

    df_raw['Proj'] = df_raw.apply(get_ui_proj, axis=1)
    pool_list = []
    for _, r in df_raw.iterrows():
        p_data = r.to_dict()
        t1, t2 = TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))
        g_id = " vs ".join(sorted([t1, t2]))
        w = weather_info.get(g_id, {'temp': '--', 'wind': 'Calm', 'condition': 'Unknown'})
        p_data.update({'Weather_Short': f"{w['temp']}°", 'Logo': get_logo_url(r['Team'])})
        p_data['Primary_Stat'] = f"WHP: {r['WHIP']:.2f}" if 'P' in str(r['POS']) else f"ISO: {r['ISO']:.3f}"
        p_data['Secondary_Stat'] = f"K/9: {r['K/9']:.1f}" if 'P' in str(r['POS']) else f"wRC: {int(r['wRC+'])}"
        w_icon = "⚪"
        wind, cond = str(w['wind']).lower(), str(w['condition']).lower()
        if "dome" in cond or "closed" in cond:
            w_icon = "🏟️"
        elif "out" in wind:
            w_icon = "🔥"
        elif "in" in wind:
            w_icon = "❄️"
        p_data['W_Icon'] = w_icon
        p_data['Adv'] = False if 'P' in str(r['POS']) else (
                    r['CleanHand'] == 'S' or (r['CleanHand'] == 'L' and p_hand_map.get(r['Opponent']) == 'R') or (
                        r['CleanHand'] == 'R' and p_hand_map.get(r['Opponent']) == 'L'))
        pool_list.append(p_data)
    unique_games = {}
    confirmed = set(df_raw[df_raw['Order'] > 0]['Team'].unique())
    for _, r in df_raw.iterrows():
        t1, t2 = TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))
        g_id = " vs ".join(sorted([t1, t2]))
        if g_id not in unique_games:
            time_d = espn_times.get(g_id, {'display': 'TBD', 'raw': datetime.max.replace(tzinfo=pytz.utc)})
            w = weather_info.get(g_id, {'temp': '--', 'condition': 'Unknown'})
            unique_games[g_id] = {'id': g_id, 'display': g_id, 'time': time_d['display'], 'sort': time_d['raw'],
                                  't1': r['Team'], 't2': r['Opponent'], 'l1': get_logo_url(r['Team']),
                                  'l2': get_logo_url(r['Opponent']), 'i1': r['Team'] in confirmed,
                                  'i2': r['Opponent'] in confirmed, 'weather': f"{w['temp']}° {w['condition']}"}
    game_list = sorted(unique_games.values(), key=lambda x: x['sort'])
    results, status = None, "SYSTEMS LIVE"
    if request.method == 'POST':
        sel_games = request.form.getlist('games')
        excluded = [g['id'] for g in game_list if g['id'] not in sel_games]
        results = run_optimizer(df_raw, num_lineups=int(request.form.get('num_lineups', 10)),
                                locks=request.form.getlist('player_locks'), stack_team=request.form.get('stack_team'),
                                min_stack=int(request.form.get('min_stack', 3)),
                                diversity=int(request.form.get('diversity', 4)),
                                exposure_limit=float(request.form.get('exposure_limit', 1.0)), excluded_games=excluded,
                                weather_data=weather_info)
        status = f"LOCKED {len(results)} LINEUPS"
    return render_template_string(HTML_BODY, results=results, status=status,
                                  teams=sorted(df_raw['Team'].dropna().unique()), games=game_list, pool=pool_list)


def get_mlb_weather_data():
    weather_map = {}
    api_to_abbr = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
                   "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
                   "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET",
                   "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA",
                   "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
                   "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
                   "Oakland Athletics": "OAK", "Athletics": "OAK", "Philadelphia Phillies": "PHI",
                   "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
                   "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
                   "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS"}
    try:
        now = datetime.now(pytz.timezone('US/Eastern'))
        games = statsapi.schedule(date=now.strftime('%Y-%m-%d'))
        for g in games:
            a, h = api_to_abbr.get(g['away_name']), api_to_abbr.get(g['home_name'])
            if not a or not h: continue
            gid = " vs ".join(sorted([a, h]))
            try:
                det = statsapi.get('game', {'gamePk': g['game_id']})
                w = det.get('gameData', {}).get('weather', {})
                weather_map[gid] = {'temp': int(w.get('temp', 70)) if str(w.get('temp')).isdigit() else 70,
                                    'wind': w.get('wind', '0 mph, Calm'), 'condition': w.get('condition', 'Clear')}
            except:
                weather_map[gid] = {'temp': 70, 'wind': '0 mph, Calm', 'condition': 'Unknown'}
    except:
        pass
    return weather_map


def get_espn_game_times():
    now = datetime.now(pytz.timezone('US/Eastern'))
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={now.strftime('%Y%m%d')}"
    times = {}
    try:
        data = requests.get(url).json()
        for e in data.get('events', []):
            teams = [TEAM_MAP.get(t['team']['abbreviation'], t['team']['abbreviation']) for t in
                     e['competitions'][0]['competitors']]
            gid = " vs ".join(sorted(teams))
            utc = datetime.strptime(e['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=pytz.utc)
            times[gid] = {'display': utc.astimezone(pytz.timezone('US/Eastern')).strftime('%I:%M %p'), 'raw': utc}
    except:
        pass
    return times


# --- HTML TEMPLATE ---
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

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB DFS Optimizer | Betify Sports</title>
    <meta name="description" content="Free MLB DFS Lineup Optimizer with real-time weather, stats, and salary data. Built for high-stakes DFS players.">
    <link rel="canonical" href="https://betifysports.com/" />

    <meta property="og:title" content="Betify Pro MLB Optimizer">
    <meta property="og:description" content="Generate winning MLB lineups with advanced stat blending and weather logic.">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://betifysports.com">

    <style>
        :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --accent: #3fb950; --text: #c9d1d9; --header: #161b22; --win: #238636; --row-alt: #1c2128; --logo-bg: rgba(255, 255, 255, 0.12); }
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
                <div class="label-group"><small>STACK TEAM</small><select name="stack_team"><option value="None">Auto-Optimized</option>{% for team in teams %}<option value="{{ team }}">{{ team }}</option>{% endfor %}</select></div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;"><div class="label-group"><small>LINEUPS</small><input type="number" name="num_lineups" value="10"></div><div class="label-group"><small>STACK SIZE</small><input type="number" name="min_stack" value="3"></div></div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;"><div class="label-group"><small>DIVERSITY</small><input type="number" name="diversity" value="4"></div><div class="label-group"><small>EXPOSURE</small><input type="number" step="0.1" name="exposure_limit" value="1.0"></div></div>
                <button type="submit" class="lock-btn">LOCK LINEUPS</button>
            </div>
            <div class="card">
                <h3>Games</h3>
                {% for g in games %}<div style="background:var(--row-alt); padding:10px; border-radius:6px; margin-bottom:8px;"><div style="display:flex; justify-content:space-between; font-size:0.75em; color:var(--accent);"><span>{{ g.time }}</span><input type="checkbox" name="games" value="{{ g.id }}" checked></div><div style="display:flex; justify-content:space-between; align-items:center;"><span><span class="status-dot {{ 'on' if g.i1 else 'off' }}"></span> {{ g.t1 }}</span><span style="color:var(--border)">vs</span><span>{{ g.t2 }} <span class="status-dot {{ 'on' if g.i2 else 'off' }}"></span></span></div><div style="font-size: 0.7em; color: #8b949e; margin-top: 4px;">{{ g.weather }}</div></div>{% endfor %}
            </div>
        </div>
        <div class="card" style="padding: 0;">
            <div class="pool-container">
                <table>
                    <thead><tr><th>LCK</th><th>PLAYER</th><th>POS</th><th>ORD</th><th>TEAM</th><th>WTH</th><th>ADV STATS</th><th>BATS</th><th>CHALK</th><th>SAL</th><th>PRJ</th></tr></thead>
                    <tbody>
                        {% for p in pool %}
                        <tr>
                            <td><input type="checkbox" name="player_locks" value="{{ p.Player }}"></td>
                            <td><img src="{{ p.Logo }}" class="team-logo-icon"><b>{{ p.Player }}</b></td>
                            <td style="color:var(--accent); font-weight:bold;">{{ p.POS }}</td>
                            <td>{{ p.Order if p.Order > 0 else '—' }}</td>
                            <td>{{ p.Team }}</td>
                            <td style="text-align:center;"><div style="font-size: 1.2em;">{{ p.W_Icon }}</div><div style="font-weight: bold;">{{ p.Weather_Short }}</div></td>
                            <td>{{ p.Primary_Stat }}<br>{{ p.Secondary_Stat }}</td>
                            <td><span class="hand-tag {{ 'adv-match' if p.Adv else 'neut-match' }}">{{ p.Hand }}</span></td>
                            <td class="stat-val" style="color:var(--accent)">{{ p.Chalk_Quality if 'P' in p.POS else p.Edge_Value }}</td>
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
    {% if results %}<div class="lineup-grid">{% for lineup in results %}<div class="card" style="border-top: 3px solid var(--accent);"><div style="display:flex; justify-content:space-between;"><b>LINEUP #{{ loop.index }}</b><span style="color:var(--accent)">{{ lineup.total_projection }} PTS</span></div><table style="margin-top:10px;">{% for p in lineup.players %}<tr><td style="color:var(--accent); font-weight:bold;">{{ p.Slot }}</td><td>{{ p.Name }} ({{ p.Team }})</td><td style="text-align:right;">${{ "{:,.0f}".format(p.Salary) }}</td></tr>{% endfor %}</table></div>{% endfor %}</div>{% endif %}
</div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)