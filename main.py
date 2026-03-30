import pandas as pd
import pulp
import re
import requests
from flask import Flask, render_template_string, request
from datetime import datetime
import pytz

app = Flask(__name__)

# --- CONFIG ---
SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"
POS_ORDER = {'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4, '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9}

# Translation for ESPN vs Your Sheet Abbreviations
# Added ATH for Athletics as per your requirement
TEAM_MAP = {
    "CHW": "CWS",
    "WSH": "WAS",
    "OAK": "ATH",  # Critical: Maps ESPN's OAK to your ATH
    "SF": "SFO",
    "AZ": "ARI",
    "TB": "TB",
    "KC": "KC",
    "TOR": "TOR",
    "COL": "COL",
    "ATL": "ATL"
}


def get_today_slate():
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    try:
        data = requests.get(url).json()
        slate = {}
        for event in data.get('events', []):
            competitors = event['competitions'][0]['competitors']

            # 1. Map them IMMEDIATELY (turns OAK into ATH)
            mapped_names = []
            for t in competitors:
                raw_abbr = t['team']['abbreviation']
                mapped_names.append(TEAM_MAP.get(raw_abbr, raw_abbr))

            # 2. Sort them AFTER mapping (ensures 'ATH vs TOR' order)
            mapped_names.sort()
            game_id = " vs ".join(mapped_names)

            # 3. Status and Time
            status_type = event['status']['type']['name']
            is_active = status_type != 'STATUS_SCHEDULED'

            utc_time = datetime.strptime(event['date'], "%Y-%m-%dT%H:%MZ")
            utc_time = pytz.utc.localize(utc_time)
            et_time = utc_time.astimezone(pytz.timezone('US/Eastern')).strftime('%I:%M %p')

            slate[game_id] = {
                'time': et_time,
                't1_locked': is_active,
                't2_locked': is_active
            }
        return slate
    except Exception as e:
        print(f"Error: {e}")
        return {}

def clean_name(name):
    if not isinstance(name, str): return ""
    name = name.upper().strip().replace(".", "").replace("'", "")
    name = re.sub(r'[•\-].*', '', name).strip()
    return name

def run_optimizer(df, num_lineups=1, locks=[], stack_team=None, min_stack=3, diversity=4, excluded_games=[]):
    all_results = []
    past_solutions = []
    pitchers_df = df[df['POS'].str.contains('P', na=False)]
    p_hand_map = pitchers_df.set_index('Team')['Hand'].to_dict()

    if excluded_games:
        for gid in excluded_games:
            teams_in_game = gid.split(" vs ")
            df = df[~df['Team'].isin(teams_in_game)].copy()

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"MLB_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        slots = list(POS_ORDER.keys())
        x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")

        prob += pulp.lpSum([df.loc[p, 'Proj'] * x[p][s] for p in players for s in slots])
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

        for t in df['Team'].unique():
            h_idx = df[(df['Team'] == t) & (~df['POS'].str.contains('P', na=False))].index.tolist()
            prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) <= 5
            if stack_team == t:
                prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= min_stack

        for sol in past_solutions:
            prob += pulp.lpSum([x[p][s] for p in sol for s in slots]) <= (len(slots) - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup_data, player_indices = [], []
            total_sal, total_proj = 0, 0
            for p in players:
                for s in slots:
                    if x[p][s].varValue == 1:
                        total_sal += df.loc[p, 'Salary']
                        total_proj += df.loc[p, 'Proj']
                        opp_p_hand = "—" if s.startswith('P') else p_hand_map.get(df.loc[p, 'Opponent'], '?')
                        lineup_data.append({
                            'Slot': s, 'Name': df.loc[p, 'Player'], 'Team': df.loc[p, 'Team'],
                            'Hand': df.loc[p, 'Hand'], 'OppP': opp_p_hand,
                            'Order': int(df.loc[p, 'Batting Order']) if df.loc[p, 'Batting Order'] > 0 else '—',
                            'Proj': round(df.loc[p, 'Proj'], 2), 'Salary': df.loc[p, 'Salary'], 'SortKey': POS_ORDER[s]
                        })
                        player_indices.append(p)
            lineup_data.sort(key=lambda x: x['SortKey'])
            all_results.append(
                {'players': lineup_data, 'total_salary': total_sal, 'total_projection': round(total_proj, 2)})
            past_solutions.append(player_indices)
        else:
            break
    return all_results

@app.route('/', methods=['GET', 'POST'])
def index():
    live_slate = get_today_slate()
    df_raw = pd.read_csv(SALARY_CSV)
    df_raw['Salary'] = pd.to_numeric(df_raw['Salary'].astype(str).replace(r'[\$,]', '', regex=True).apply(
        lambda x: float(x.replace('k', '')) * 1000 if 'k' in x else x), errors='coerce').fillna(0)
    df_raw['Batting Order'] = pd.to_numeric(df_raw['Batting Order'], errors='coerce').fillna(0)
    df_raw['Proj'] = pd.to_numeric(df_raw['Projected Points'], errors='coerce').fillna(0)

    p_hand_map = df_raw[df_raw['POS'].str.contains('P', na=False)].set_index('Team')['Hand'].to_dict()
    pool_list = [{**r.to_dict(), 'OppP': "—" if 'P' in str(r['POS']) else p_hand_map.get(r['Opponent'], '?')} for _, r
                 in df_raw.iterrows()]

    # Generate game list with Corrected Indicators
    games = sorted(list(set([" vs ".join(sorted([str(r['Team']), str(r['Opponent'])])) for _, r in df_raw.iterrows()])))
    available_games = []
    for g in games:
        game_info = live_slate.get(g, {'time': 'TBD', 't1_locked': False, 't2_locked': False})

        # Logic for individual Checks and X's
        t1_icon = '<span style="color:#00ff41;">✔</span>' if game_info['t1_locked'] else '<span style="color:#ff4b4b;">✘</span>'
        t2_icon = '<span style="color:#00ff41;">✔</span>' if game_info['t2_locked'] else '<span style="color:#ff4b4b;">✘</span>'
        indicator = f"{t1_icon}{t2_icon}"

        available_games.append({
            "id": g,
            "display": f"{g} ({game_info['time']})",
            "indicator": indicator
        })

    results, status = None, "Ready."
    if request.method == 'POST':
        results = run_optimizer(df_raw, num_lineups=int(request.form.get('num_lineups', 5)),
                                locks=request.form.getlist('player_locks'),
                                stack_team=request.form.get('stack_team'),
                                diversity=int(request.form.get('diversity', 4)),
                                excluded_games=[g['id'] for g in available_games if
                                                g['id'] not in request.form.getlist('games')])
        status = f"Generated {len(results)} lineups."

    return render_template_string(HTML_BODY, results=results, status=status,
                                  teams=sorted(df_raw['Team'].dropna().unique()),
                                  games=available_games, pool=pool_list)

HTML_BODY = """
<body style="background:#0e1117; color:white; font-family:sans-serif; padding:30px;">
    <h1 style="color:#00ff41;">BETIFY MLB PRO</h1>
    <div style="background:#1c2128; padding:15px; border-left:5px solid #00ff41; margin-bottom:25px; font-family:monospace;">STATUS: {{ status }}</div>

    <form method="post">
        <div style="background:#161b22; padding:25px; border-radius:10px; border:1px solid #30363d; display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
            <div>
                <label>Lineups / Diversity:</label>
                <div style="display:flex; gap:10px; margin-top:5px;">
                    <input type="number" name="num_lineups" value="5" style="width:50%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px;">
                    <input type="number" name="diversity" value="4" style="width:50%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px;">
                </div>
                <br><label>Stack Team:</label>
                <select name="stack_team" style="width:100%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px; margin-top:5px;">
                    <option value="None">Auto</option>
                    {% for team in teams %}<option value="{{ team }}">{{ team }}</option>{% endfor %}
                </select>
            </div>
            <div style="background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d;">
                <label style="color:#00ff41; font-weight:bold;">Active Slate & Lineups:</label>
                <div style="max-height:150px; overflow-y:auto; margin-top:10px;">
                    {% for game in games %}
                    <label style="display:flex; justify-content:space-between; align-items:center; font-size:0.85em; margin-bottom:8px; padding-right:10px;">
                        <span><input type="checkbox" name="games" value="{{ game.id }}" checked> {{ game.display }}</span>
                        <b style="letter-spacing: 3px; font-family: 'Segoe UI Symbol', sans-serif;">{{ game.indicator|safe }}</b>
                    </label>
                    {% endfor %}
                </div>
            </div>
            <button type="submit" style="grid-column: span 2; background:#238636; color:white; padding:15px; border:none; border-radius:6px; font-weight:bold; cursor:pointer;">GENERATE</button>
        </div>

        <details style="background:#161b22; border:1px solid #30363d; border-radius:8px; margin-bottom:20px;">
            <summary style="padding:15px; cursor:pointer; font-weight:bold; color:#00ff41;">+ MANAGE PLAYER POOL & LOCKS</summary>
            <div style="padding:15px; max-height:400px; overflow-y:auto;">
                <table style="width:100%; border-collapse:collapse; font-size:0.85em;">
                    <thead style="position:sticky; top:0; background:#161b22; color:#8b949e; text-align:left;">
                        <tr><th>LOCK</th><th>PLAYER</th><th>POS</th><th>TEAM</th><th>ORD</th><th>HAND</th><th>vs P</th><th>SALARY</th><th>PROJ</th></tr>
                    </thead>
                    <tbody>
                        {% for p in pool %}
                        <tr style="border-bottom:1px solid #21262d;">
                            <td><input type="checkbox" name="player_locks" value="{{ p.Player }}"></td>
                            <td><b>{{ p.Player }}</b></td><td>{{ p.POS }}</td><td>{{ p.Team }}</td>
                            <td style="color:#58a6ff;">{{ p['Batting Order']|int if p['Batting Order'] > 0 else '—' }}</td>
                            <td>{{ p.Hand }}</td>
                            <td style="font-weight:bold; color:{% if p.OppP != '—' and p.Hand != p.OppP and p.OppP != '?' %}#00ff41{% else %}#8b949e{% endif %};">{{ p.OppP }}</td>
                            <td>${{ "{:,.0f}".format(p.Salary) }}</td><td style="color:#00ff41;">{{ p.Proj|round(1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </details>
    </form>

    {% if results %}{% for lineup in results %}
    <div style="margin-top:20px; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px;">
        <div style="display:flex; justify-content:space-between; border-bottom:2px solid #30363d; padding-bottom:10px; margin-bottom:15px;">
            <h3 style="color:#00ff41; margin:0;">Lineup #{{ loop.index }}</h3>
            <div style="font-family:monospace;">SAL: ${{ "{:,.0f}".format(lineup.total_salary) }} | PROJ: {{ lineup.total_projection }}</div>
        </div>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
            <tr style="text-align:left; color:#8b949e;"><th>POS</th><th>PLAYER</th><th>ORD</th><th>HAND</th><th>vs P</th><th>TEAM</th><th>SAL</th><th>PROJ</th></tr>
            {% for p in lineup.players %}
            <tr style="border-bottom:1px solid #21262d;">
                <td style="padding:8px; font-weight:bold; color:#8b949e;">{{p.Slot.replace('1','').replace('2','').replace('3','')}}</td>
                <td><b>{{p.Name}}</b></td><td>{{p.Order}}</td><td>{{p.Hand}}</td>
                <td style="font-weight:bold; color:{% if p.OppP != '—' and p.Hand != p.OppP and p.OppP != '?' %}#00ff41{% else %}#8b949e{% endif %};">{{p.OppP}}</td>
                <td>{{p.Team}}</td><td>${{ "{:,.0f}".format(p.Salary) }}</td><td style="color:#00ff41;">{{p.Proj}}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endfor %}{% endif %}
</body>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)