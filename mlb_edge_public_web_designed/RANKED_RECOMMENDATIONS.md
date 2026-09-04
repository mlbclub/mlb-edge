# Ranked recommendations and loading changes

At the user's request, the public board now ranks every available priced market
by its unmodified model hit probability, takes one selection per game, and shows
up to five games. EV only breaks exact probability ties. There is no minimum hit
probability, edge or EV gate; even a negative-EV selection can appear. The earlier
value-qualified list and strict-filter functions remain for historical diagnostics,
but the board consumes all candidates. Missing/invalid prices are not fabricated.
Totals are ranked by unconditional hit probability, including the chance of a push.
This is a recommendation-policy change, not a measured increase in model accuracy.

V10's calibrated development result remains 349/544 high-confidence games (64.15%).
This does not establish the prospective accuracy or profitability of the top-five
policy. No probabilities were raised and no model retraining was needed here.

Loading changes:
- Model and team-data objects are reused until their artifact revision changes.
- Weather/park requests are skipped when neither loaded estimator uses those inputs.
  Context-consuming legacy bundles still request them.
- Display summaries reuse the feature row already calculated for prediction.
- The date-query button no longer clears every visitor's five-minute data cache.

Verification: 27 tests pass. On a real historical row, ten repeated display-summary
calculations fell from 0.334 seconds to below 0.0001 seconds with identical values.
This is a component benchmark, not a claimed end-to-end site loading time.
Live network latency and Streamlit cold starts still affect initial loading.
