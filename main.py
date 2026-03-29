import pandas as pd
import pulp
import re
from flask import Flask, render_template_string, request

app = Flask(__name__)

# --- CONFIG ---
SALARY_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"

POS_ORDER = {'P1': 0, 'P2': 1, 'C': 2, '1B': 3, '2B': 4, '3B': 5, 'SS': 6, 'OF1': 7, 'OF2': 8, 'OF3': 9}


def clean_name(name):
    if not isinstance(name, str): return ""
    name = name.upper().strip()
    name = re.sub(r'\.', '', name)
    suffixes = [' JR', ' SR', ' II', ' III', ' IV']
    for s in suffixes:
        if name.endswith(s): name = name[:-len(s)]
    return name.strip()


def get_games(df):
    """Creates unique game IDs using 'Team' and 'Opponent' columns."""
    if 'Team' not in df.columns or 'Opponent' not in df.columns:
        return []

    # We create a sorted tuple so NYY vs BOS and BOS vs NYY are the same game
    matchups = set()
    for _, row in df.iterrows():
        t1, t2 = str(row['Team']), str(row['Opponent'])
        game_id = " vs ".join(sorted([t1, t2]))
        matchups.add(game_id)

    return [{"id": g} for g in sorted(list(matchups))]


def run_optimizer(df, num_lineups=1, locks=[], excludes=[], stack_team=None, min_stack=3, diversity=4,
                  excluded_games=[]):
    all_results = []
    past_solutions = []

    # 1. Game Filtering Logic
    if excluded_games:
        # If a game like "BOS vs NYY" is excluded, remove players from BOTH teams
        for game_id in excluded_games:
            teams_in_game = game_id.split(" vs ")
            df = df[~df['Team'].isin(teams_in_game)].copy()

    df['CleanPlayer'] = df['Player'].apply(clean_name)
    df = df[~df['CleanPlayer'].isin([clean_name(e) for e in excludes])].copy()
    teams = df['Team'].unique().tolist()

    for i in range(num_lineups):
        prob = pulp.LpProblem(f"MLB_Opt_{i}", pulp.LpMaximize)
        players = df.index.tolist()
        slots = list(POS_ORDER.keys())
        x = pulp.LpVariable.dicts("x", (players, slots), cat="Binary")

        prob += pulp.lpSum([df.loc[p, 'Proj'] * x[p][s] for p in players for s in slots])

        # Core Constraints
        prob += pulp.lpSum([df.loc[p, 'Salary'] * x[p][s] for p in players for s in slots]) <= 50000
        for s in slots: prob += pulp.lpSum([x[p][s] for p in players]) == 1
        for p in players: prob += pulp.lpSum([x[p][s] for s in slots]) <= 1

        # Position Eligibility
        for p in players:
            pos = str(df.loc[p, 'POS'])
            for s in slots:
                if s.startswith('P') and 'P' not in pos:
                    prob += x[p][s] == 0
                elif s == 'C' and 'C' not in pos:
                    prob += x[p][s] == 0
                elif s == '1B' and '1B' not in pos:
                    prob += x[p][s] == 0
                elif s == '2B' and '2B' not in pos:
                    prob += x[p][s] == 0
                elif s == '3B' and '3B' not in pos:
                    prob += x[p][s] == 0
                elif s == 'SS' and 'SS' not in pos:
                    prob += x[p][s] == 0
                elif s.startswith('OF') and 'OF' not in pos:
                    prob += x[p][s] == 0

        # Team Limits (Max 5 Batters)
        for t in teams:
            h_idx = df[(df['Team'] == t) & (~df['POS'].str.contains('P', na=False))].index.tolist()
            prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) <= 5

        # Locks
        clean_locks = [clean_name(l) for l in locks]
        for p in players:
            if df.loc[p, 'CleanPlayer'] in clean_locks:
                prob += pulp.lpSum([x[p][s] for s in slots]) == 1

        # Hitter-Only Stack
        if stack_team and stack_team != "None":
            h_idx = df[(df['Team'] == stack_team) & (~df['POS'].str.contains('P', na=False))].index.tolist()
            prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= min_stack
        else:
            t_vars = pulp.LpVariable.dicts("tstack", teams, cat="Binary")
            for t in teams:
                h_idx = df[(df['Team'] == t) & (~df['POS'].str.contains('P', na=False))].index.tolist()
                prob += pulp.lpSum([x[p][s] for p in h_idx for s in slots]) >= min_stack * t_vars[t]
            prob += pulp.lpSum([t_vars[t] for t in teams]) >= 1

        # Diversity Requirement
        for sol in past_solutions:
            prob += pulp.lpSum([x[p][s] for p in sol for s in slots]) <= (len(slots) - diversity)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup_data = []
            player_indices = []
            total_sal, total_proj = 0, 0
            for p in players:
                for s in slots:
                    if x[p][s].varValue == 1:
                        total_sal += df.loc[p, 'Salary']
                        total_proj += df.loc[p, 'Proj']
                        lineup_data.append({
                            'Slot': s, 'Name': df.loc[p, 'Player'], 'Team': df.loc[p, 'Team'],
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
    df_raw = pd.read_csv(SALARY_CSV)
    available_teams = sorted(df_raw['Team'].dropna().unique().tolist())
    available_games = get_games(df_raw)

    results, status = None, "Ready."

    if request.method == 'POST':
        try:
            num_l = int(request.form.get('num_lineups', 5))
            div = int(request.form.get('diversity', 4))
            target_team = request.form.get('stack_team')
            locks = [n.strip() for n in request.form.get('locks', '').split(',') if n.strip()]

            selected_game_ids = request.form.getlist('games')
            all_game_ids = [g['id'] for g in available_games]
            excluded_games = [gid for gid in all_game_ids if gid not in selected_game_ids]

            df = df_raw.copy()
            df['Salary'] = df['Salary'].astype(str).replace(r'[\$,]', '', regex=True)
            df['Salary'] = df['Salary'].apply(lambda x: float(x.replace('k', '')) * 1000 if 'k' in x else float(x))
            df['Proj'] = pd.to_numeric(df['Projected Points'], errors='coerce').fillna(0)

            results = run_optimizer(df, num_lineups=num_l, locks=locks, stack_team=target_team, diversity=div,
                                    excluded_games=excluded_games)
            status = f"Generated {len(results)} lineups."
        except Exception as e:
            status = f"Error: {str(e)}"

    return render_template_string(HTML_BODY, results=results, status=status, teams=available_teams,
                                  games=available_games)


HTML_BODY = """
<body style="background:#0e1117; color:white; font-family:sans-serif; padding:30px;">
    <h1 style="color:#00ff41; margin-bottom:5px;">BETIFY MLB PRO</h1>
    <div style="background:#1c2128; padding:15px; border-left:5px solid #00ff41; margin-bottom:25px; font-family:monospace;">
        STATUS: <span style="color:#58a6ff;">{{ status }}</span>
    </div>

    <form method="post" style="background:#161b22; padding:25px; border-radius:10px; border:1px solid #30363d; max-width:850px;">
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
            <div>
                <label>Lineups / Diversity:</label>
                <div style="display:flex; gap:10px;">
                    <input type="number" name="num_lineups" value="5" style="width:50%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px;">
                    <input type="number" name="diversity" value="4" style="width:50%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px;">
                </div>
                <br>
                <label>Forced Hitter Stack:</label>
                <select name="stack_team" style="width:100%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px;">
                    <option value="None">Auto-Stack</option>
                    {% for team in teams %}
                    <option value="{{ team }}">{{ team }}</option>
                    {% endfor %}
                </select>
            </div>

            <div style="background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d;">
                <label style="color:#00ff41; font-weight:bold;">Active Slate:</label>
                <div style="max-height:150px; overflow-y:auto; margin-top:10px;">
                    {% for game in games %}
                    <label style="display:block; margin-bottom:5px; font-size:0.85em;">
                        <input type="checkbox" name="games" value="{{ game.id }}" checked> {{ game.id }}
                    </label>
                    {% endfor %}
                </div>
            </div>
        </div>

        <label>Locks:</label>
        <input type="text" name="locks" placeholder="Nolan McLean, etc." style="width:100%; background:#0d1117; color:white; border:1px solid #30363d; padding:8px; margin-bottom:20px;">

        <button type="submit" style="width:100%; background:#238636; color:white; padding:15px; border:none; border-radius:6px; font-weight:bold; cursor:pointer;">GENERATE LINEUPS</button>
    </form>

    {% if results %}
        {% for lineup in results %}
        <div style="margin-top:20px; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px;">
            <div style="display:flex; justify-content:space-between; border-bottom:2px solid #30363d; padding-bottom:10px; margin-bottom:15px;">
                <h3 style="color:#00ff41; margin:0;">Lineup #{{ loop.index }}</h3>
                <div style="font-family:monospace;">SAL: ${{ "{:,.0f}".format(lineup.total_salary) }} | PROJ: {{ lineup.total_projection }}</div>
            </div>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="text-align:left; color:#8b949e; font-size:0.9em;"><th>POS</th><th>PLAYER</th><th>TEAM</th><th>SAL</th><th>PROJ</th></tr>
                {% for p in lineup.players %}
                <tr style="border-bottom:1px solid #21262d;">
                    <td style="padding:8px; font-weight:bold; color:#8b949e;">{{p.Slot.replace('1','').replace('2','').replace('3','')}}</td>
                    <td><b>{{p.Name}}</b></td><td>{{p.Team}}</td><td>${{ "{:,.0f}".format(p.Salary) }}</td><td style="color:#00ff41;">{{p.Proj}}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endfor %}
    {% endif %}
</body>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)