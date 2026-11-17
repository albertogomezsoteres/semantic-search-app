# Retrieval evaluation

8821 shots. Queries are written the way a fan might type them (33 dev, 22 test). A result counts as relevant when the shot's metadata matches the query's rule (see `eval/queries.yaml`). The configuration is chosen on dev: the best nDCG@10 among those averaging at most 300 ms per query. Test was written before any tuning and is only reported. Latency is per query on a laptop CPU (embed + search + rerank).

## Dev queries

| model | template | reranker | p@10 | r@10 | mrr | ndcg@10 | latency_ms |
|---|---|---|---|---|---|---|---|
| intfloat/e5-small-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.197 | 0.022 | 0.275 | 0.211 | 156.341 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.173 | 0.019 | 0.271 | 0.183 | 106.659 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | none | 0.127 | 0.015 | 0.259 | 0.142 | 7.467 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.115 | 0.008 | 0.221 | 0.133 | 137.696 |
| intfloat/e5-small-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.112 | 0.011 | 0.222 | 0.130 | 209.767 |
| intfloat/e5-small-v2 | moment | none | 0.130 | 0.025 | 0.170 | 0.127 | 14.188 |
| intfloat/e5-small-v2 | fluent | none | 0.085 | 0.027 | 0.191 | 0.096 | 13.620 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | none | 0.064 | 0.003 | 0.157 | 0.075 | 7.010 |
| random baseline | - | - | 0.009 | 0.001 | 0.022 | 0.009 | 0.000 |

## Test queries

| model | template | reranker | p@10 | r@10 | mrr | ndcg@10 | latency_ms |
|---|---|---|---|---|---|---|---|
| intfloat/e5-small-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.336 | 0.102 | 0.508 | 0.395 | 95.259 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.300 | 0.100 | 0.432 | 0.354 | 109.315 |
| intfloat/e5-small-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.286 | 0.052 | 0.441 | 0.328 | 145.165 |
| intfloat/e5-small-v2 | moment | none | 0.277 | 0.097 | 0.405 | 0.319 | 8.648 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | none | 0.218 | 0.056 | 0.375 | 0.258 | 7.024 |
| intfloat/e5-small-v2 | fluent | none | 0.191 | 0.060 | 0.377 | 0.227 | 8.562 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.200 | 0.011 | 0.300 | 0.219 | 137.506 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | none | 0.082 | 0.004 | 0.168 | 0.087 | 6.567 |
| random baseline | - | - | 0.014 | 0.001 | 0.040 | 0.014 | 0.000 |

Chosen on dev: **intfloat/e5-small-v2**, **moment** template, reranker **cross-encoder/ms-marco-MiniLM-L6-v2**.

## Per-query results (intfloat/e5-small-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2)

| split | query | n_relevant | p@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| dev | nodded in from a corner | 59 | 0.300 | 1.000 | 0.415 |
| dev | spot kick that the keeper stopped | 71 | 0.100 | 0.143 | 0.073 |
| dev | screamer from way outside the area that went in | 52 | 0.000 | 0.000 | 0.000 |
| dev | cheeky lob over the goalie | 52 | 0.100 | 0.100 | 0.064 |
| dev | bicycle kick attempt | 43 | 0.600 | 0.500 | 0.522 |
| dev | volleyed home after a ball into the box from the wing | 36 | 0.000 | 0.000 | 0.000 |
| dev | headed goal during the extra thirty minutes | 6 | 0.000 | 0.000 | 0.000 |
| dev | fast break goal in the closing minutes | 22 | 0.000 | 0.000 | 0.000 |
| dev | unbelievable miss from a few yards out | 237 | 0.000 | 0.000 | 0.000 |
| dev | shot rattled the crossbar | 208 | 0.000 | 0.056 | 0.000 |
| dev | player failed to convert in the shootout | 64 | 0.000 | 0.000 | 0.000 |
| dev | curled a set piece straight in | 12 | 0.000 | 0.000 | 0.000 |
| dev | late leveller to rescue a draw | 33 | 0.000 | 0.000 | 0.000 |
| dev | last-gasp match winner in added time | 21 | 0.000 | 0.000 | 0.000 |
| dev | followed up and tucked it in after the keeper parried | 72 | 0.000 | 0.000 | 0.000 |
| dev | slotted home after a defence-splitting pass | 69 | 0.000 | 0.026 | 0.000 |
| dev | finish from a pull-back from the byline | 24 | 0.000 | 0.000 | 0.000 |
| dev | flying header | 44 | 0.000 | 0.000 | 0.000 |
| dev | flick with the heel | 40 | 0.000 | 0.000 | 0.000 |
| dev | volley from distance | 497 | 1.000 | 1.000 | 1.000 |
| dev | goalkeeper wins the one against one duel | 135 | 0.500 | 1.000 | 0.649 |
| dev | early goal in the opening minutes | 103 | 1.000 | 1.000 | 1.000 |
| dev | left foot rocket from range into the net | 37 | 0.800 | 1.000 | 0.870 |
| dev | bundled in off the chest or knee | 54 | 0.000 | 0.056 | 0.000 |
| dev | goal that cut the deficit | 52 | 0.000 | 0.000 | 0.000 |
| dev | skied effort nowhere near the goal | 533 | 0.600 | 1.000 | 0.707 |
| dev | rolled into an empty net | 82 | 0.000 | 0.020 | 0.000 |
| dev | converted penalty during normal play | 95 | 0.500 | 1.000 | 0.649 |
| dev | header from a free-kick delivery that went in | 14 | 0.000 | 0.030 | 0.000 |
| dev | goal in the tournament final | 30 | 0.000 | 0.077 | 0.000 |
| dev | shot off target from long distance under pressure | 137 | 1.000 | 1.000 | 1.000 |
| dev | goal straight after a throw-in | 1 | 0.000 | 0.000 | 0.000 |
| dev | Messi spot kick | 8 | 0.000 | 0.077 | 0.000 |
| test | towering header that hit the back of the net | 165 | 1.000 | 1.000 | 1.000 |
| test | penalty blazed over or wide | 32 | 0.000 | 0.033 | 0.000 |
| test | keeper saves a long-distance drive | 456 | 1.000 | 1.000 | 1.000 |
| test | tap-in from right in front of goal | 189 | 0.000 | 0.000 | 0.000 |
| test | goal to make it all square | 133 | 0.000 | 0.000 | 0.000 |
| test | breakaway goal after winning the ball back | 56 | 0.000 | 0.067 | 0.000 |
| test | half-volley charged down by a defender | 315 | 0.900 | 1.000 | 0.922 |
| test | dead-ball strike kept out by the keeper | 69 | 0.000 | 0.038 | 0.000 |
| test | goal from a ball whipped in from the flank | 148 | 0.000 | 0.020 | 0.000 |
| test | header that hit the frame of the goal | 50 | 0.500 | 1.000 | 0.649 |
| test | goal just before half-time | 26 | 0.000 | 0.000 | 0.000 |
| test | goal that made it a two-goal cushion or more | 289 | 0.100 | 1.000 | 0.220 |
| test | successful kick in the shootout | 128 | 0.700 | 0.500 | 0.607 |
| test | attempt after a corner routine that the keeper held | 235 | 1.000 | 1.000 | 1.000 |
| test | glorious heading chance wasted | 32 | 0.100 | 0.500 | 0.139 |
| test | shot after beating a man with a dribble | 9 | 0.000 | 0.000 | 0.000 |
| test | shot took a nick off a defender and went in | 38 | 0.000 | 0.021 | 0.000 |
| test | volley that the goalkeeper kept out | 275 | 1.000 | 1.000 | 1.000 |
| test | right-footer from just outside the area into the bottom corner | 5 | 0.000 | 0.000 | 0.000 |
| test | goal scored in the extra period | 19 | 0.500 | 1.000 | 0.614 |
| test | Mbappé scoring in the final | 5 | 0.300 | 1.000 | 0.606 |
| test | Richarlison acrobatic finish | 3 | 0.300 | 1.000 | 0.933 |
