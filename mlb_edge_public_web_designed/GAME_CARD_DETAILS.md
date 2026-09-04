# Pregame card details

Each team now has a compact panel beneath its name: recent five wins/losses
(green win, red loss; newest at right), team batting average and bullpen ERA
over the last five games, starter season ERA and last-five-start ERA, and its
last ten home games for the home team or away games for the away team.
All statistics exclude the displayed game's start time and later records.
ERA uses 9 × total earned runs / total innings; batting average uses hits / at-bats.
Starter season means the current calendar season across teams, not career ERA.
Missing or zero-innings records display an em dash, not a zero ERA.

Key relievers are explicitly defined as the team's top three pitchers by saves
plus holds in its preceding 30 games, with positive totals. This is a statistical
proxy, not an official managerial role or an availability guarantee. Each pitcher's
streak counts consecutive TEAM GAMES with an appearance, backwards from the team's
most recent game. It is not calendar-day consecutive pitching. A collection marker
for every team/game distinguishes complete games from missing bullpen records.
Incomplete coverage displays a collection status rather than invented zeroes.

`enrich_games` extracts appearances from existing boxscore fetches and writes
`data/reliever_appearances.csv`. The workflow commits the file; site requests only
read it, with no extra per-game network calls. Artifact revision invalidates caches.

Starter names use a local Korean display dictionary, with original English in
parentheses and larger type. The current 32 scheduled starters plus common starters
are mapped. Unmapped future names show a Korean-name-pending label plus the original
name rather than an invented translation; extend `card_view.PITCHER_KO` as needed.

Validation: 31 tests, including weighted ERA, future-record exclusion, venue split,
individual streaks, collection completeness, Korean diacritic normalization and
HTML escaping. The actual rendered card was checked locally using current game data.
The prediction models and recommendation ranking are unchanged by this display work.
