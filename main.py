import pandas as pd
import pulp
import re
import requests
from flask import Flask, render_template_string, request
from datetime import datetime
import pytz
from pybaseball import batting_stats, pitching_stats

app = Flask(__name__)

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

def clean_hand_str(h):
    if pd.isna(h): return "?"
    s = str(h).upper()
    if 'R' in s: return 'R'
    if 'L' in s: return 'L'
    if 'S' in s: return 'S'
    return "?"

def get_today_slate():
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    try:
        data = requests.get(url).json()
        slate = {}
        for event in data.get('events', []):
            competitors = event['competitions'][0]['competitors']
            raw_names = [t['team']['abbreviation'] for t in competitors]
            mapped_names = sorted([TEAM_MAP.get(name, name) for name in raw_names])
            game_id = " vs ".join(mapped_names)
            status_type = event['status']['type']['name']
            is_active = status_type != 'STATUS_SCHEDULED'
            utc_time = datetime.strptime(event['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=pytz.utc)
            et_time = utc_time.astimezone(pytz.timezone('US/Eastern')).strftime('%I:%M %p')
            slate[game_id] = {'time': et_time, 't1_locked': is_active, 't2_locked': is_active}
        return slate
    except:
        return {}

def get_weighted_stats():
    try:
        h26 = batting_stats(2026); h25 = batting_stats(2025)
        h_merge = pd.merge(h26, h25, on='Name', how='outer', suffixes=('_26', '_25')).fillna(0)
        h_merge['W_Barrel'] = (h_merge['Barrel%_26'] * 0.7) + (h_merge['Barrel%_25'] * 0.3)
        h_merge['W_xwOBA'] = (h_merge['xwOBA_26'] * 0.7) + (h_merge['xwOBA_25'] * 0.3)
        h_merge['W_BB'] = (h_merge['BB%_26'] * 0.7) + (h_merge['BB%_25'] * 0.3)
        h_merge['Edge_Value'] = (h_merge['W_Barrel'] * 40) + (h_merge['W_xwOBA'] * 40) + (h_merge['W_BB'] * 20)
        h_final = h_merge[['Name', 'Edge_Value', 'W_Barrel', 'W_xwOBA']].rename(columns={'W_Barrel': 'Barrel%', 'W_xwOBA': 'xwOBA'})

        p26 = pitching_stats(2026); p25 = pitching_stats(2025)
        p_merge = pd.merge(p26, p25, on='Name', how='outer', suffixes=('_26', '_25')).fillna(0)
        p_merge['W_SIERA'] = (p_merge['SIERA_26'] * 0.7) + (p_merge['SIERA_25'] * 0.3)
        p_merge['W_KBB'] = (p_merge['K-BB%_26'] * 0.7) + (p_merge['K-BB%_25'] * 0.3)
        p_merge['W_SwStr'] = (p_merge['SwStr%_26'] * 0.7) + (p_merge['SwStr%_25'] * 0.3)
        p_merge['Chalk_Quality'] = (p_merge['W_KBB'] * 100) + (10 / p_merge['W_SIERA'].replace(0, 5)) + (p_merge['W_SwStr'] * 100)
        p_final = p_merge[['Name', 'Chalk_Quality', 'W_SIERA', 'W_KBB']].rename(columns={'W_SIERA': 'SIERA', 'W_KBB': 'K-BB%'})
        return h_final, p_final
    except:
        return pd.DataFrame(), pd.DataFrame()

def run_optimizer(df_input, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4, excluded_games=[]):
    df = df_input.copy()
    all_results = []
    used_player_indices = []

    # 1. Filter excluded games
    if excluded_games:
        for gid in excluded_games:
            teams_in_game = gid.split(" vs ")
            df = df[~df['Team'].isin(teams_in_game)]

    # 2. Rebuild Pitcher Hand Map for Platoon Adv
    pitchers_df = df[df['POS'].str.contains('P', na=False)]
    p_hand_map = pitchers_df.set_index('Team')['CleanHand'].to_dict()

    def apply_logic(row):
        base_proj = row['Proj']
        if 'P' in str(row['POS']):
            return base_proj * (1 + (row.get('Chalk_Quality', 0) / 400)) if row.get('Chalk_Quality', 0) > 0 else base_proj
        if row.get('Edge_Value', 0) > 0:
            base_proj = (base_proj * 0.85) + (row['Edge_Value'] * 0.15)
        # Platoon Adv (12%)
        b_clean, o_clean = row['CleanHand'], p_hand_map.get(row['Opponent'], '?')
        if b_clean == 'S' or (b_clean == 'L' and o_clean == 'R') or (b_clean == 'R' and o_clean == 'L'):
            base_proj *= 1.12
        return base_proj

    df['Solver_Proj'] = df.apply(apply_logic, axis=1)
    teams_to_stack = [stack_team] if (stack_team and stack_team != "None") else df['Team'].unique()

    for i in range(num_lineups):
        best_lineup_for_pass = None
        highest_score = -1

        for current_team in teams_to_stack:
            prob = pulp.LpProblem(f"MLB_Gen_{i}_{current_team}", pulp.LpMaximize)
            players = df.index.tolist()
            slots = list(POS_ORDER.keys())
            x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")

            prob += pulp.lpSum([df.loc[p, 'Solver_Proj'] * x[p][s] for p in players for s in slots])
            prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000

            for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
            for p in players: prob += pulp.lpSum([x[p][s] for s in slots]) <= 1
            for p in players:
                if df.loc[p, 'Player'] in locks: prob += pulp.lpSum([x[p][s] for s in slots]) == 1

            for p in players:
                pos = str(df.loc[p, 'POS'])
                for s in slots:
                    if (s.startswith('P') and 'P' not in pos) or (s == 'C' and 'C' not in pos) or \
                       (s == '1B' and '1B' not in pos) or (s == '2B' and '2B' not in pos) or \
                       (s == '3B' and '3B' not in pos) or (s == 'SS' and 'SS' not in pos) or \
                       (s.startswith('OF') and 'OF' not in pos): prob += x[p][s] == 0

            for past_indices in used_player_indices:
                prob += pulp.lpSum([x[p][s] for p in past_indices for s in slots]) <= (len(slots) - diversity)

            h_idx = df[(df['Team'] == current_team) & (~df['POS'].str.contains('P', na=False))].index.tolist()
            if len(h_idx) >= int(min_stack):
                prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= int(min_stack)
            else: continue

            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if pulp.LpStatus[prob.status] == 'Optimal':
                score = pulp.value(prob.objective)
                if score > highest_score:
                    highest_score = score
                    lineup_data, p_indices = [], []
                    total_sal, total_proj = 0, 0
                    for p in players:
                        for s in slots:
                            if x[p][s].varValue == 1:
                                row = df.loc[p]
                                b_h, o_h = row['CleanHand'], p_hand_map.get(row['Opponent'], '?')
                                lineup_data.append({
                                    'Slot': s, 'Name': row['Player'], 'Team': row['Team'], 'Hand': row['Hand'],
                                    'OppP': "—" if s.startswith('P') else o_h,
                                    'Adv': False if s.startswith('P') else (b_h == 'S' or (b_h == 'L' and o_h == 'R') or (b_h == 'R' and o_h == 'L')),
                                    'Proj': round(row['Proj'], 2), 'Salary': row['Salary'], 'SortKey': POS_ORDER[s]
                                })
                                p_indices.append(p)
                                total_sal += row['Salary']
                                total_proj += row['Proj']
                    lineup_data.sort(key=lambda x: x['SortKey'])
                    best_lineup_for_pass = {'players': lineup_data, 'total_salary': total_sal, 'total_projection': round(total_proj, 2), 'indices': p_indices}

        if best_lineup_for_pass:
            all_results.append(best_lineup_for_pass)
            used_player_indices.append(best_lineup_for_pass['indices'])
        else: break
    return all_results

@app.route('/', methods=['GET', 'POST'])
def index():
    live_slate = get_today_slate()
    h_fg, p_fg = get_weighted_stats()
    df_raw = pd.read_csv(SALARY_CSV)

    df_raw['CleanHand'] = df_raw['Hand'].apply(clean_hand_str)
    df_raw['Salary'] = pd.to_numeric(df_raw['Salary'].astype(str).replace(r'[\$,]', '', regex=True).apply(
        lambda x: float(x.replace('k', '')) * 1000 if 'k' in x else x), errors='coerce').fillna(0)
    df_raw['Proj'] = pd.to_numeric(df_raw['Projected Points'], errors='coerce').fillna(0)

    if not h_fg.empty: df_raw = df_raw.merge(h_fg, left_on='Player', right_on='Name', how='left').fillna(0)
    if not p_fg.empty: df_raw = df_raw.merge(p_fg, left_on='Player', right_on='Name', how='left').fillna(0)

    pitchers_df = df_raw[df_raw['POS'].str.contains('P', na=False)]
    p_hand_map = pitchers_df.set_index('Team')['CleanHand'].to_dict()

    pool_list, unique_games = [], set()
    for _, r in df_raw.iterrows():
        p_data = r.to_dict()
        b_clean, o_clean = r['CleanHand'], p_hand_map.get(r['Opponent'], '?')
        p_data['OppP'] = "—" if 'P' in str(r['POS']) else o_clean
        p_data['Adv'] = False if 'P' in str(r['POS']) else (b_clean == 'S' or (b_clean == 'L' and o_clean == 'R') or (b_clean == 'R' and o_clean == 'L'))
        pool_list.append(p_data)
        t1, t2 = TEAM_MAP.get(str(r['Team']), str(r['Team'])), TEAM_MAP.get(str(r['Opponent']), str(r['Opponent']))
        unique_games.add(" vs ".join(sorted([t1, t2])))

    available_games = []
    for g in sorted(list(unique_games)):
        game_info = live_slate.get(g, {'time': 'TBD', 't1_locked': False, 't2_locked': False})
        t1_icon = '<span style="color:#00ff41;">✔</span>' if game_info['t1_locked'] else '<span style="color:#ff4b4b;">✘</span>'
        t2_icon = '<span style="color:#00ff41;">✔</span>' if game_info['t2_locked'] else '<span style="color:#ff4b4b;">✘</span>'
        available_games.append({"id": g, "display": f"{g} ({game_info['time']})", "indicator": f"{t1_icon}{t2_icon}"})

    results, status = None, "Ready."
    if request.method == 'POST':
        results = run_optimizer(df_raw, num_lineups=int(request.form.get('num_lineups', 5)),
                                locks=request.form.getlist('player_locks'),
                                stack_team=request.form.get('stack_team'),
                                min_stack=request.form.get('min_stack', 3),
                                diversity=int(request.form.get('diversity', 4)),
                                excluded_games=[g['id'] for g in available_games if g['id'] not in request.form.getlist('games')])
        status = f"Generated {len(results)} lineups."

    return render_template_string(HTML_BODY, results=results, status=status, teams=sorted(df_raw['Team'].dropna().unique()), games=available_games, pool=pool_list)

HTML_BODY = """
<!DOCTYPE html>
<html>
<head>
    <style>
        :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --accent: #00ff41; --text: #c9d1d9; --muted: #8b949e; --edge: #f1e05a; --chalk: #58a6ff; }
        body { background: var(--bg); color: var(--text); font-family: -apple-system, sans-serif; padding: 20px; line-height: 1.5; }
        .container { max-width: 1200px; margin: auto; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 20px; }
        .status-bar { background: var(--card); border-left: 4px solid var(--accent); padding: 10px 20px; font-family: monospace; font-size: 0.9em; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        input, select, button { background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 8px; border-radius: 6px; }
        button { background: #238636; border: none; font-weight: bold; cursor: pointer; color: white; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.82em; table-layout: fixed; }
        th { text-align: left; color: var(--muted); padding: 12px 8px; border-bottom: 1px solid var(--border); cursor: pointer; }
        th:hover { color: var(--accent); background: #21262d; }
        td { padding: 10px 8px; border-bottom: 1px solid #21262d; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .sticky-head { position: sticky; top: 0; background: var(--card); z-index: 10; }
        .hand-adv { background: #238636; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
        .hand-neut { background: #30363d; color: #c9d1d9; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
        .edge-col { color: var(--edge); font-weight: bold; }
        .chalk-col { color: var(--chalk); font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <div class="header"><h1 style="color: var(--accent); margin:0;">BETIFY MLB PRO + WEIGHTED EDGE</h1></div>
    <div class="status-bar">SYSTEM STATUS: {{ status }}</div>
    <form method="post">
        <div class="grid">
            <div class="card">
                <h3>Settings</h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div>Lineups: <input type="number" name="num_lineups" value="5" style="width:100%;"></div>
                    <div>Diversity: <input type="number" name="diversity" value="4" style="width:100%;"></div>
                    <div>Stack Team: 
                        <select name="stack_team" style="width:100%;">
                            <option value="None">Auto-Select Best</option>
                            {% for team in teams %}<option value="{{ team }}">{{ team }}</option>{% endfor %}
                        </select>
                    </div>
                    <div>Min Stack Size: <input type="number" name="min_stack" value="3" min="0" max="5" style="width:100%;"></div>
                </div>
                <br>
                <button type="submit" style="width:100%; padding:12px;">GENERATE LINEUPS</button>
            </div>
            <div class="card">
                <h3>Slate</h3>
                <div style="max-height:150px; overflow-y:auto; font-size:0.85em;">
                    {% for game in games %}
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span><input type="checkbox" name="games" value="{{ game.id }}" checked> {{ game.display }}</span>
                        <b>{{ game.indicator|safe }}</b>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <h3 style="margin:0;">Player Pool</h3>
                <input type="text" id="playerSearch" placeholder="Filter..." style="max-width:200px;">
            </div>
            <div style="max-height:500px; overflow-y:auto; border:1px solid var(--border); border-radius:6px;">
                <table>
                    <colgroup>
                        <col style="width: 40px;"><col style="width: 140px;"><col style="width: 50px;"><col style="width: 50px;">
                        <col style="width: 60px;"><col style="width: 80px;"><col style="width: 80px;"><col style="width: 80px;">
                        <col style="width: 80px;"><col style="width: 55px;">
                    </colgroup>
                    <thead>
                        <tr class="sticky-head">
                            <th>LCK</th><th>PLAYER</th><th>POS</th><th>TEAM</th><th>BATS</th>
                            <th onclick="sortTable(5, true)">BRL% / KBB</th>
                            <th onclick="sortTable(6, true)">xwOBA / SIERA</th>
                            <th onclick="sortTable(7, true)">SUBSTANCE</th>
                            <th onclick="sortTable(8, true)">SALARY</th>
                            <th onclick="sortTable(9, true)">PROJ</th>
                        </tr>
                    </thead>
                    <tbody id="playerBody">
                        {% for p in pool %}
                        <tr>
                            <td><input type="checkbox" name="player_locks" value="{{ p.Player }}"></td>
                            <td><b>{{ p.Player }}</b></td>
                            <td>{{ p.POS }}</td>
                            <td>{{ p.Team }}</td>
                            <td><span class="{{ 'hand-adv' if p.Adv else 'hand-neut' }}">{{ p.Hand }}</span></td>
                            <td class="{{ 'chalk-col' if 'P' in p.POS else 'edge-col' }}">
                                {{ (p['K-BB%']*100)|round(1) if 'P' in p.POS else (p['Barrel%']*100)|round(1) }}%
                            </td>
                            <td class="{{ 'chalk-col' if 'P' in p.POS else 'edge-col' }}">
                                {{ p.SIERA|round(2) if 'P' in p.POS else p.xwOBA|round(3) }}
                            </td>
                            <td style="font-weight:bold;">{{ p.Chalk_Quality|round(1) if 'P' in p.POS else p.Edge_Value|round(1) }}</td>
                            <td>${{ "{:,.0f}".format(p.Salary) }}</td>
                            <td style="color:var(--accent); font-weight:bold;">{{ p.Proj|round(1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </form>

    {% if results %}{% for lineup in results %}<div class="card">
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border); padding-bottom:10px; margin-bottom:10px;">
            <span style="font-weight:bold;">Lineup #{{ loop.index }}</span>
            <span style="color:var(--accent); font-weight:bold;">SAL: ${{ "{:,.0f}".format(lineup.total_salary) }} | PROJ: {{ lineup.total_projection }}</span>
        </div>
        <table>
            <tr style="color:var(--muted); font-size:0.85em; text-transform:uppercase;">
                <th>POS</th><th>PLAYER</th><th>TEAM</th><th>BATS</th><th>VS P</th><th>SAL</th><th>PROJ</th>
            </tr>
            {% for p in lineup.players %}<tr>
                <td style="width:50px; color:var(--muted);">{{p.Slot.replace('1','').replace('2','').replace('3','')}}</td>
                <td><b>{{p.Name}}</b></td><td>{{p.Team}}</td>
                <td><span class="{{ 'hand-adv' if p.Adv else 'hand-neut' }}">{{p.Hand}}</span></td>
                <td>{{p.OppP}}</td><td>${{ "{:,.0f}".format(p.Salary) }}</td><td style="color:var(--accent);">{{p.Proj}}</td>
            </tr>{% endfor %}
        </table>
    </div>{% endfor %}{% endif %}
</div>
<script>
function sortTable(colIdx, isNum = false) {
    const tbody = document.getElementById("playerBody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const sorted = rows.sort((a, b) => {
        let x = a.children[colIdx].innerText, y = b.children[colIdx].innerText;
        if (isNum) { x = parseFloat(x.replace(/[^0-9.-]+/g, "")) || 0; y = parseFloat(y.replace(/[^0-9.-]+/g, "")) || 0; }
        return y - x;
    });
    tbody.innerHTML = ""; sorted.forEach(row => tbody.appendChild(row));
}
document.getElementById('playerSearch').addEventListener('input', function() {
    let q = this.value.toLowerCase();
    document.querySelectorAll('#playerBody tr').forEach(r => { r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none'; });
});
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)