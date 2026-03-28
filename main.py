from flask import Flask, render_template_string, request, send_from_directory
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import requests
import io
import re
import os
from datetime import datetime
from pybaseball import batting_stats
from sqlalchemy import create_engine

app = Flask(__name__)

# --- CONFIG & RAILWAY DB SETUP ---
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzCRSTDnslz-zmGESH1CFhjsYD7NJa8yHkapMFu1JIR0M1PQDwZzMIDCmhPBUNU6kzLJy8-3_ioR4Y/pub?gid=1189680617&single=true&output=csv"
DK_ORDER = {'P1': 1, 'P2': 2, 'C': 3, '1B': 4, '2B': 5, '3B': 6, 'SS': 7, 'OF1': 8, 'OF2': 9, 'OF3': 10}

# Railway uses DATABASE_URL. We convert 'postgres://' to 'postgresql://' for SQLAlchemy compatibility.
DB_URL = os.environ.get('DATABASE_URL')
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DB_URL) if DB_URL else None


# --- DATA HELPERS ---
def find_col(df, options):
    for opt in options:
        if opt in df.columns: return opt
    return None


def get_cached_stats():
    """Fetches stats from Postgres; falls back to scraping if DB is empty/missing."""
    table_name = "mlb_stats_cache"

    # Attempt to read from DB first to prevent data loss
    if engine:
        try:
            return pd.read_sql(table_name, engine)
        except:
            pass

    try:
        # Pull fresh data (2025 and 2026)
        s25 = batting_stats(2025)[['Name', 'wRC+', 'Barrel%', 'HardHit%']]
        s26 = batting_stats(2026)[['Name', 'wRC+', 'Barrel%', 'xwOBA']]
        s25.columns = ['Name', 'wRC25', 'Bar25', 'HH25']
        s26.columns = ['Name', 'wRC26', 'Bar26', 'xwOBA26']
        master = pd.merge(s26, s25, on='Name', how='outer').fillna(0)

        # Persistent save to Railway Postgres
        if engine:
            master.to_sql(table_name, engine, if_exists='replace', index=False)

        return master
    except Exception as e:
        print(f"Scrape Error: {e}")
        return pd.DataFrame(columns=['Name', 'wRC26', 'Bar26', 'wRC25'])


def clean_num(v):
    if pd.isna(v) or v == "": return 0.0
    s = str(v).replace('$', '').replace(',', '').lower()
    if 'k' in s:
        try:
            return float(s.replace('k', '')) * 1000
        except:
            return 0.0
    try:
        return float(re.sub(r'[^\d.]', '', s))
    except:
        return 0.0


def get_hand(row):
    try:
        val = str(row.iloc[16]).strip().upper() if len(row) > 16 else "R"
        clean = re.sub(r'[^LRS]', '', val)
        return clean if clean in ['L', 'R', 'S'] else "R"
    except:
        return "R"


# --- OPTIMIZER LOGIC ---
def run_strategic_optimizer(df, options, locks, excludes):
    p_df = df[df['POS'].str.contains('P', na=False)].copy().sort_values('Raw_Proj', ascending=False)
    chalk_pitchers = p_df.head(3)['Player'].tolist()
    p_hands = {row['Team']: row['Hand_Clean'] for _, row in p_df.iterrows()}

    def weight(r):
        proj = r['Raw_Proj']
        if 'P' in str(r['POS']): return proj * 1.20 if r['Player'] in chalk_pitchers else proj
        p_hand = p_hands.get(r['Opponent'])
        if p_hand:
            is_adv = (r['Hand_Clean'] == 'S') or (r['Hand_Clean'] != p_hand)
            proj *= 1.15 if is_adv else 0.85
        if r['wRC25'] > 125 and r['wRC26'] < 100: proj *= 1.10
        if r['Bar26'] > 12.0: proj *= 1.12
        return proj

    df['Strat_Proj'] = df.apply(weight, axis=1)
    df_pool = df[~df['Player'].isin(excludes)].copy()
    df_pool = df_pool[(df_pool['Salary_Clean'] > 0) & (df_pool['Strat_Proj'] > 0)].copy()

    players, slots = df_pool.index.tolist(), list(DK_ORDER.keys())
    hit_slots = [s for s in slots if 'P' not in s]
    num_lineups = int(options.get('num_lineups', 1))
    max_overlap = int(options.get('max_overlap', 7))

    all_lineups, prev_indices = [], []
    for i in range(num_lineups):
        prob = LpProblem(f"MLB_Opto_{i}", LpMaximize)
        x = LpVariable.dicts(f"x", (players, slots), cat="Binary")
        prob += lpSum([df_pool.loc[p, 'Strat_Proj'] * x[p][s] for p in players for s in slots])
        for s in slots: prob += lpSum([x[p][s] for p in players]) == 1
        for p in players: prob += lpSum([x[p][s] for s in slots]) <= 1
        prob += lpSum([df_pool.loc[p, 'Salary_Clean'] * x[p][s] for p in players for s in slots]) <= 50000

        for p in players:
            p_pos = str(df_pool.loc[p, 'POS'])
            if df_pool.loc[p, 'Player'] in locks: prob += lpSum([x[p][s] for s in slots]) == 1
            for s in slots:
                if re.sub(r'\d+', '', s) not in p_pos: prob += x[p][s] == 0

        teams = df_pool['Team'].unique()
        stack_vars = LpVariable.dicts(f"stk", teams, cat="Binary")
        prob += lpSum([stack_vars[t] for t in teams]) == 1
        for t in teams:
            t_hits = [p for p in players if df_pool.loc[p, 'Team'] == t and 'P' not in str(df_pool.loc[p, 'POS'])]
            prob += lpSum([x[p][s] for p in t_hits for s in hit_slots]) >= 4 * stack_vars[t]

        for prev in prev_indices:
            prob += lpSum([x[p][s] for p in prev for s in slots]) <= max_overlap

        prob.solve()
        if LpStatus[prob.status] != 'Optimal': break

        res, curr_idx = [], []
        for p in players:
            for s in slots:
                if x[p][s].varValue == 1:
                    curr_idx.append(p)
                    row = df_pool.loc[p].to_dict()
                    row['Slot'], row['DK_Rank'] = s, DK_ORDER[s]
                    res.append(row)
        prev_indices.append(curr_idx)
        all_lineups.append(pd.DataFrame(res).sort_values('DK_Rank'))
    return all_lineups


# --- HTML TEMPLATE (Preserved from Source) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>MLB Strategic Engine</title>
    <style>
        :root { --bg: #000; --panel: #121212; --accent: #00ff41; --text: #fff; --sub: #bbb; --blue: #38b6ff; --red: #ff3131; --gold: #ffde59; --drawer-w: 92vw; }
        body { font-family: -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 0; overflow-x: hidden; }
        .header { background: #1a1a1a; padding: 15px; border-bottom: 2px solid var(--accent); text-align: center; position: sticky; top: 0; z-index: 1100; }
        .scroller-container { background: #1a1a1a; display: flex; align-items: center; border-bottom: 1px solid #333; padding: 10px 5px; }
        .toggle-all-btn { background: #333; color: #fff; border: 1px solid #555; border-radius: 6px; padding: 10px; font-size: 10px; font-weight: 900; margin-right: 10px; cursor: pointer; white-space: nowrap; }
        .game-picker { display: flex; overflow-x: auto; gap: 8px; white-space: nowrap; -webkit-overflow-scrolling: touch; }
        .game-chip { display: inline-block; padding: 10px 16px; border-radius: 8px; border: 1px solid #444; background: #0a0a0a; font-size: 11px; font-weight: 900; cursor: pointer; color: #777; transition: 0.2s; }
        .game-chip.active { border-color: var(--accent); color: var(--accent); background: #002208; }
        .game-chip input { display: none; }
        .settings-bar { background: #222; padding: 12px; display: flex; justify-content: center; gap: 20px; border-bottom: 1px solid #333; font-size: 13px; font-weight: bold; }
        .settings-bar input { background: #000; color: var(--accent); border: 1px solid var(--accent); padding: 5px; border-radius: 4px; text-align: center; width: 45px; font-weight: 900; }
        .btn-gen { background: var(--accent); color: #000; border: none; padding: 18px; border-radius: 8px; width: 95%; font-weight: 900; font-size: 18px; margin: 10px auto; display: block; text-transform: uppercase; cursor: pointer; }
        .container { padding: 10px; max-width: 800px; margin: auto; }
        .search-box { width: 100%; padding: 16px; margin-bottom: 15px; background: #111; border: 2px solid #444; color: #fff; border-radius: 8px; font-size: 18px; box-sizing: border-box; outline: none; }
        .player-card { background: var(--panel); border: 1px solid #333; border-radius: 12px; margin-bottom: 12px; padding: 18px; }
        .player-name { font-weight: 900; font-size: 18px; }
        .proj-val { font-weight: 900; color: var(--accent); font-size: 24px; }
        .control-bar { display: flex; gap: 12px; margin-top: 15px; }
        .btn-toggle { flex: 1; border: 2px solid #444; background: #1a1a1a; color: #fff; padding: 14px; border-radius: 8px; font-weight: 800; text-align: center; font-size: 14px; }
        input[type="checkbox"]:checked + .btn-toggle.lock { background: var(--accent); color: #000; border-color: var(--accent); }
        input[type="checkbox"]:checked + .btn-toggle.out { background: var(--red); color: #fff; border-color: var(--red); }
        #drawer { position: fixed; top: 0; right: -100%; width: var(--drawer-w); height: 100%; background: #000; border-left: 3px solid var(--accent); z-index: 2000; transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1); overflow-y: auto; padding: 20px; box-sizing: border-box; }
        #drawer.open { right: 0; }
        .overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); display: none; z-index: 1500; }
        .overlay.active { display: block; }
        .fab { position: fixed; bottom: 25px; right: 25px; width: 65px; height: 65px; background: var(--accent); color: #000; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 30px; z-index: 1400; box-shadow: 0 0 15px var(--accent); }
        .hand-L { color: var(--red); } .hand-R { color: var(--blue); } .hand-S { color: var(--gold); }
    </style>
</head>
<body>
    <form method="post" id="mainForm">
    <div class="header"><span style="font-weight: 900; letter-spacing: 2px; color: var(--accent); font-size: 20px;">STRATEGIC ENGINE</span></div>
    <div class="scroller-container">
        <button type="button" class="toggle-all-btn" onclick="toggleAllGames()">SELECT ALL</button>
        <div class="game-picker" id="gamePicker">
            {% for game in all_games %}
            <label class="game-chip {% if game in selected_games %}active{% endif %}">
                <input type="checkbox" name="selected_games" value="{{game}}" {% if game in selected_games %}checked{% endif %} class="game-check" onchange="document.getElementById('mainForm').submit()">
                {{ game }}
            </label>
            {% endfor %}
        </div>
    </div>
    <div class="settings-bar">
        <span>LINEUPS: <input type="number" name="num_lineups" value="{{ num_lineups|default(1) }}"></span>
        <span>OVERLAP: <input type="number" name="max_overlap" value="{{ max_overlap|default(7) }}"></span>
    </div>
    <button type="submit" class="btn-gen">GENERATE LINEUPS</button>
    <div class="container">
        <input type="text" id="pSearch" class="search-box" placeholder="SEARCH POOL..." onkeyup="filter()">
        <div id="playerList">
            {% for i, r in df.sort_values('Raw_Proj', ascending=False).iterrows() %}
            <div class="player-card" data-s="{{r.Player}} {{r.Team}} {{r.POS}}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 2;">
                        <div class="player-name">{{r.Player}} <span class="hand-{{r.Hand_Clean}}">({{r.Hand_Clean}})</span></div>
                        <div style="color:var(--sub); font-size:13px; margin-top:4px;">{{r.POS}} | {{r.Team}} vs {{r.Opponent}} (#{{r.Order}})</div>
                    </div>
                    <div style="text-align: right; flex: 1;">
                        <div class="proj-val">{{r.Raw_Proj}}</div>
                        <div style="color:var(--blue); font-weight:900;">${{ "{:,.0f}".format(r.Salary_Clean) }}</div>
                    </div>
                </div>
                <div class="control-bar">
                    <label style="flex:1;"><input type="checkbox" name="lock" value="{{r.Player}}" class="lock-check" {% if r.Player in locks %}checked{% endif %}><div class="btn-toggle lock">LOCK</div></label>
                    <label style="flex:1;"><input type="checkbox" name="exclude" value="{{r.Player}}" class="exclude-check" {% if r.Player in excludes %}checked{% endif %}><div class="btn-toggle out">OUT</div></label>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    <div class="fab" onclick="toggleDrawer()">📋</div>
    <div id="overlay" class="overlay" onclick="toggleDrawer()"></div>
    <div id="drawer" class="{% if lineups %}open{% endif %}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: var(--accent); font-weight: 900;">LINEUPS</h2>
            <button type="button" onclick="toggleDrawer()" style="background:#fff; color:#000; border:none; padding:10px 20px; border-radius:5px; font-weight:900;">CLOSE</button>
        </div>
        {% for lineup in lineups %}
        <div style="background:#111; border:1px solid #444; border-radius:12px; padding:15px; margin-bottom:20px;">
            <div style="font-weight:900; border-bottom:2px solid var(--accent); padding-bottom:8px; margin-bottom:10px; display:flex; justify-content:space-between;">
                <span>#{{ loop.index }}</span>
                <span style="color:var(--blue);">${{ "{:,.0f}".format(lineup.Salary_Clean.sum()) }}</span>
            </div>
            <table style="width:100%; border-collapse:collapse;">
                {% for _, r in lineup.iterrows() %}
                <tr>
                    <td style="color:#888; font-weight:bold; font-size:11px; width:40px;">{{r.Slot}}</td>
                    <td style="padding:8px 0;"><div style="font-weight:900; font-size:14px;">{{r.Player}}</div><div style="font-size:10px; color:var(--sub);">{{r.Team}} (#{{r.Order}})</div></td>
                    <td style="text-align:right; font-weight:900; color:var(--accent);">{{r.Strat_Proj|round(1)}}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endfor %}
    </div>
    </form>
    <script>
        function toggleDrawer() { document.getElementById('drawer').classList.toggle('open'); document.getElementById('overlay').classList.toggle('active'); }
        function toggleAllGames() {
            let checks = document.querySelectorAll('.game-check');
            let anyUnchecked = Array.from(checks).some(c => !c.checked);
            checks.forEach(c => c.checked = anyUnchecked);
            document.getElementById('mainForm').submit();
        }
        function filter() {
            let q = document.getElementById('pSearch').value.toUpperCase();
            let cards = document.getElementsByClassName('player-card');
            for (let c of cards) { c.style.display = c.getAttribute('data-s').toUpperCase().includes(q) ? "" : "none"; }
        }
        window.onload = function() { {% if lineups %} document.getElementById('overlay').classList.add('active'); {% endif %} }
    </script>
</body>
</html>
"""


@app.route('/favicon.ico')
def favicon(): return '', 204


@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        resp = requests.get(CSV_URL)
        df = pd.read_csv(io.StringIO(resp.text))

        # Mapping
        game_col = find_col(df, ['Game Info', 'Game_Info', 'Matchup', 'Game'])
        name_col = find_col(df, ['Player', 'Name', 'Name '])
        sal_col = find_col(df, ['Salary', 'Salary '])
        proj_col = find_col(df, ['Projected Points', 'AvgPointsPerGame', 'Points'])
        pos_col = find_col(df, ['POS', 'Position', 'Roster Position'])
        team_col = find_col(df, ['Team', 'TeamAbbrev'])
        opp_col = find_col(df, ['Opponent', 'Opp Abbrev'])
        ord_col = find_col(df, ['Order', 'Projected_Order', 'Batting Order'])

        # Data Prep
        clean_df = pd.DataFrame()
        clean_df['Player'] = df[name_col] if name_col else "Unknown"
        clean_df['Raw_Proj'] = df[proj_col].apply(clean_num) if proj_col else 0.0
        clean_df['Salary_Clean'] = df[sal_col].apply(clean_num) if sal_col else 0.0
        clean_df['POS'] = df[pos_col].fillna("UTL") if pos_col else "UTL"
        clean_df['Team'] = df[team_col].fillna("??") if team_col else "??"
        clean_df['Opponent'] = df[opp_col].fillna("??") if opp_col else "??"
        clean_df['Game Info'] = df[game_col].fillna("Main Slate") if game_col else "Main Slate"
        clean_df['Order'] = pd.to_numeric(df[ord_col], errors='coerce').fillna(0).astype(int) if ord_col else 0
        clean_df['Hand_Clean'] = df.apply(get_hand, axis=1)

        all_games = sorted(clean_df['Game Info'].unique().tolist())
        selected_games = request.form.getlist('selected_games') or all_games

        filtered_df = clean_df[clean_df['Game Info'].isin(selected_games)].copy()

        # Stats Logic (Now persistent via Postgres)
        stats = get_cached_stats()
        final_df = pd.merge(filtered_df, stats, left_on='Player', right_on='Name', how='left').fillna(0)

        lineups, locks, excludes = [], request.form.getlist('lock'), request.form.getlist('exclude')
        num_lineups, max_overlap = request.form.get('num_lineups', 1), request.form.get('max_overlap', 7)

        if request.method == 'POST' and 'num_lineups' in request.form:
            lineups = run_strategic_optimizer(final_df, request.form, locks, excludes)

        return render_template_string(HTML_TEMPLATE, df=final_df, lineups=lineups, locks=locks, excludes=excludes,
                                      num_lineups=num_lineups, max_overlap=max_overlap,
                                      all_games=all_games, selected_games=selected_games)
    except Exception as e:
        return f"System Error: {str(e)}"


if __name__ == '__main__':
    # Railway environment handles PORT automatically
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))