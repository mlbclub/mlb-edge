from mlb_model.odds import summarize_event
from mlb_model.probability import market_probabilities

event={
 "id":"x","home_team":"Los Angeles Dodgers","away_team":"San Diego Padres","commence_time":"2026-09-03T02:00:00Z",
 "bookmakers":[
  {"title":"BookA","markets":[
   {"key":"h2h","outcomes":[{"name":"Los Angeles Dodgers","price":1.65},{"name":"San Diego Padres","price":2.35}]},
   {"key":"totals","outcomes":[{"name":"Over","point":8.5,"price":1.91},{"name":"Under","point":8.5,"price":1.95}]},
   {"key":"spreads","outcomes":[{"name":"Los Angeles Dodgers","point":-1.5,"price":2.10},{"name":"San Diego Padres","point":1.5,"price":1.78}]},
  ]},
  {"title":"BookB","markets":[
   {"key":"h2h","outcomes":[{"name":"Los Angeles Dodgers","price":1.67},{"name":"San Diego Padres","price":2.30}]},
   {"key":"totals","outcomes":[{"name":"Over","point":8.5,"price":1.93},{"name":"Under","point":8.5,"price":1.93}]},
   {"key":"spreads","outcomes":[{"name":"Los Angeles Dodgers","point":-1.5,"price":2.12},{"name":"San Diego Padres","point":1.5,"price":1.76}]},
  ]},
 ]
}
s=summarize_event(event)
assert s["total_line"]==8.5
assert s["home_minus_1_5_odds"]==2.12
assert 0 < s["home_market_novig"] < 1
p=market_probabilities(4.8,4.1,8.5)
assert 0 < p["home_minus_1_5"] < 1
assert abs(p["over_prob"]+p["under_prob"]+p["push_prob"]-1)<1e-6
print("V3 odds/probability smoke test OK")
