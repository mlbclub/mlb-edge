# Bullpen verification and wider recommendations

Compared all 30 teams' former 30-game saves-plus-holds top-three proxy against
the Closer Monkey depth chart dated September 3, 2026 (published September 4).
Membership differs for 16 teams. The proxy can omit newly assigned closers and
include pitchers whose earlier holds no longer describe their current role.
For example, San Diego now includes Mason Miller; the old list omitted him.
All 90 depth-chart names matched pitcher IDs on MLB's active roster snapshot.
The before/after comparison is reproducible with `python audit_bullpen_roles.py`.

The daily workflow refreshes `bullpen_roles.csv` from the latest dated chart linked
on the publisher's home page, then cross-checks all 30 official MLB active rosters.
Failure or incomplete data does not overwrite the last successful file. The live
page makes no new roster/network calls. Checked snapshots must precede the game
and be at most three days old; charts older than seven days are rejected during
refresh. Unknown/inactive players are not silently substituted. Historical games
without an eligible snapshot explicitly show an unverified statistical estimate.
Depth-chart backup order is labelled as setup candidates and committee closers
are distinguished. This is an independently sourced role estimate, not a manager's
official designation or confirmation that a pitcher can pitch today.

Individual consecutive-team-game appearance counts still come from MLB boxscores.
Changing the role list does not change a pitcher's actual usage counts. Snapshots,
source URLs and check timestamps are stored, and changes invalidate site caches.

The recommendation default rises from five to ten games. Visitors can choose
5, 10, 15 or all games. There are still no minimum probability/edge/EV gates;
one priced selection per game is ranked using the unmodified model hit probability.
On the current 16-game saved snapshot, default selection returns 10 and all returns
16 (previous default 5). Missing prices still cannot produce executable selections.
This expands coverage without claiming to increase predictive accuracy.

Sources:
- https://closermonkey.com/2026/09/04/updated-closer-depth-chart/
- https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&date=YYYY-MM-DD
