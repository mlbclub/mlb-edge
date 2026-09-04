"""Compare statistical bullpen assignments with independently checked roles."""
import pandas as pd
from mlb_model.config import TEAM_GAMES, DATA_DIR
from mlb_model.game_details import RELIEF_FILE
from mlb_model.bullpen_roles import ROLE_FILE


def run():
    t=pd.read_csv(TEAM_GAMES);r=pd.read_csv(RELIEF_FILE);roles=pd.read_csv(ROLE_FILE)
    rows=[]
    for tid,g in roles.groupby('team_id'):
        checked=pd.to_datetime(g.checked_at.iloc[0],utc=True)
        past=t[t.team_id.eq(tid)&(pd.to_datetime(t.game_date,utc=True)<checked)].sort_values('game_date').tail(30)
        a=r[r.team_id.eq(tid)&r.game_pk.isin(past.game_pk)&r.pitcher_id.ne(0)]
        sums=a.groupby('pitcher_id')[['saves','holds']].sum().sum(axis=1)
        old_ids=sums[sums.gt(0)].sort_values(ascending=False).head(3).index
        old=[a[a.pitcher_id.eq(pid)].iloc[-1]['name'] for pid in old_ids]
        new=g[g.active_verified.eq(True)].sort_values('rank')
        rows.append(dict(team=g.team.iloc[0],old=';'.join(old),new=';'.join(new.name),
            membership_changed=set(old_ids)!=set(new.pitcher_id),
            additions=';'.join(new[~new.pitcher_id.isin(old_ids)].name),
            source=g.source.iloc[0],checked_at=checked.isoformat()))
    out=pd.DataFrame(rows);out.to_csv(DATA_DIR/'bullpen_role_audit.csv',index=False)
    print(f'{out.membership_changed.sum()}/{len(out)} team memberships differ from the statistical proxy')
    return out


if __name__=='__main__': run()
