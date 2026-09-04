# V10 site integration

Both Streamlit entry points invalidate cached predictions when the model, team
data or pick rules change. V10 moneyline probabilities flow to game cards,
moneyline details, market edge/EV calculations and recommendation candidates.

Totals retain the separate run models plus residual/market/similarity adjustment.
A display inconsistency was corrected: the final conditional over/under probabilities
are converted to unconditional hit probabilities using the push mass, and the
same values now feed both cards and recommendation hit probabilities/EV. Thus
over + under + push = 1 at integer and half-run lines. The 64.15% development
moneyline result does not describe totals or betting returns.

The existing daily workflow now generates today's KST predictions after model
training and commits `data/today_market_predictions.csv` together with the model.
The site continues to request current odds through its five-minute live cache.
No fabricated odds or forced recommendations are introduced when markets are absent.
The public site's `ODDS_API_KEY` secret must already be configured independently
of the GitHub Actions secret. Model output is available for all scheduled games;
recommendations still require the existing probability, edge and EV gates.

Regression coverage includes display/recommendation parity, push-aware EV,
all market candidate types, one recommended selection per game, missing prices,
model cache invalidation and V10 live/batch prediction parity (22 tests total).
