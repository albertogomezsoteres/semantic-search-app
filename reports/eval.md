# Retrieval evaluation

1494 shots. Queries are written the way a fan might type them (31 dev, 22 test). A result counts as relevant when the shot's metadata matches the query's rule (see `eval/queries.yaml`). The configuration is chosen on dev: the best nDCG@10 among those averaging at most 300 ms per query. Test was written before any tuning and is only reported. Latency is per query on a laptop CPU (embed + search + rerank).

Skipped (no relevant shots in this data): headed goal during the extra thirty minutes, goal straight after a throw-in

## Dev queries

| model | template | reranker | p@10 | r@10 | mrr | ndcg@10 | latency_ms |
|---|---|---|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 | fluent | BAAI/bge-reranker-base | 0.194 | 0.131 | 0.423 | 0.250 | 832.894 |
| BAAI/bge-base-en-v1.5 | moment | BAAI/bge-reranker-base | 0.206 | 0.136 | 0.300 | 0.228 | 601.410 |
| BAAI/bge-small-en-v1.5 | moment | BAAI/bge-reranker-base | 0.184 | 0.120 | 0.336 | 0.210 | 567.189 |
| intfloat/e5-small-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.181 | 0.122 | 0.279 | 0.206 | 98.017 |
| BAAI/bge-base-en-v1.5 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.171 | 0.109 | 0.306 | 0.200 | 113.438 |
| intfloat/e5-small-v2 | moment | BAAI/bge-reranker-base | 0.165 | 0.106 | 0.324 | 0.194 | 553.273 |
| intfloat/e5-small-v2 | fluent | BAAI/bge-reranker-base | 0.148 | 0.106 | 0.334 | 0.190 | 793.377 |
| BAAI/bge-base-en-v1.5 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.155 | 0.091 | 0.300 | 0.187 | 155.182 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | BAAI/bge-reranker-base | 0.139 | 0.089 | 0.286 | 0.166 | 607.457 |
| BAAI/bge-small-en-v1.5 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.148 | 0.091 | 0.237 | 0.166 | 101.489 |
| BAAI/bge-small-en-v1.5 | fluent | BAAI/bge-reranker-base | 0.136 | 0.084 | 0.272 | 0.164 | 826.523 |
| intfloat/e5-small-v2 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.145 | 0.101 | 0.262 | 0.160 | 94.007 |
| intfloat/e5-small-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.129 | 0.084 | 0.254 | 0.158 | 142.132 |
| BAAI/bge-base-en-v1.5 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.123 | 0.121 | 0.278 | 0.154 | 100.633 |
| intfloat/e5-small-v2 | terse | BAAI/bge-reranker-base | 0.132 | 0.085 | 0.292 | 0.153 | 606.502 |
| BAAI/bge-base-en-v1.5 | moment | none | 0.129 | 0.082 | 0.267 | 0.146 | 17.138 |
| BAAI/bge-base-en-v1.5 | terse | BAAI/bge-reranker-base | 0.116 | 0.105 | 0.241 | 0.145 | 611.428 |
| sentence-transformers/all-MiniLM-L6-v2 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.119 | 0.082 | 0.255 | 0.142 | 88.525 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.119 | 0.073 | 0.221 | 0.139 | 111.781 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.100 | 0.041 | 0.249 | 0.128 | 149.361 |
| sentence-transformers/all-MiniLM-L6-v2 | terse | BAAI/bge-reranker-base | 0.106 | 0.068 | 0.239 | 0.128 | 576.543 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | BAAI/bge-reranker-base | 0.100 | 0.045 | 0.236 | 0.125 | 840.303 |
| intfloat/e5-small-v2 | terse | none | 0.116 | 0.075 | 0.224 | 0.123 | 9.141 |
| BAAI/bge-small-en-v1.5 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.097 | 0.058 | 0.187 | 0.116 | 146.807 |
| BAAI/bge-base-en-v1.5 | terse | none | 0.093 | 0.100 | 0.194 | 0.112 | 16.966 |
| BAAI/bge-small-en-v1.5 | terse | BAAI/bge-reranker-base | 0.068 | 0.062 | 0.232 | 0.111 | 591.067 |
| BAAI/bge-base-en-v1.5 | fluent | none | 0.093 | 0.041 | 0.236 | 0.110 | 18.494 |
| BAAI/bge-small-en-v1.5 | moment | none | 0.097 | 0.062 | 0.155 | 0.100 | 9.442 |
| intfloat/e5-small-v2 | moment | none | 0.090 | 0.040 | 0.183 | 0.099 | 9.086 |
| intfloat/e5-small-v2 | fluent | none | 0.087 | 0.050 | 0.186 | 0.096 | 10.433 |
| BAAI/bge-small-en-v1.5 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.077 | 0.054 | 0.197 | 0.091 | 92.766 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | none | 0.071 | 0.034 | 0.178 | 0.086 | 7.296 |
| sentence-transformers/all-MiniLM-L6-v2 | terse | none | 0.081 | 0.038 | 0.157 | 0.082 | 6.952 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | none | 0.065 | 0.029 | 0.155 | 0.076 | 8.536 |
| BAAI/bge-small-en-v1.5 | fluent | none | 0.055 | 0.032 | 0.113 | 0.061 | 9.953 |
| BAAI/bge-small-en-v1.5 | terse | none | 0.035 | 0.023 | 0.107 | 0.040 | 9.359 |
| random baseline | - | - | 0.010 | 0.005 | 0.025 | 0.010 | 0.000 |

## Test queries

| model | template | reranker | p@10 | r@10 | mrr | ndcg@10 | latency_ms |
|---|---|---|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 | moment | BAAI/bge-reranker-base | 0.332 | 0.234 | 0.547 | 0.443 | 649.877 |
| intfloat/e5-small-v2 | moment | BAAI/bge-reranker-base | 0.327 | 0.200 | 0.570 | 0.421 | 562.505 |
| intfloat/e5-small-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.300 | 0.230 | 0.576 | 0.410 | 145.747 |
| BAAI/bge-base-en-v1.5 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.273 | 0.199 | 0.506 | 0.393 | 126.038 |
| BAAI/bge-small-en-v1.5 | moment | BAAI/bge-reranker-base | 0.300 | 0.175 | 0.523 | 0.385 | 580.945 |
| BAAI/bge-base-en-v1.5 | moment | none | 0.282 | 0.196 | 0.452 | 0.356 | 18.884 |
| intfloat/e5-small-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.277 | 0.175 | 0.429 | 0.355 | 98.964 |
| intfloat/e5-small-v2 | fluent | BAAI/bge-reranker-base | 0.291 | 0.200 | 0.436 | 0.355 | 809.297 |
| BAAI/bge-base-en-v1.5 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.264 | 0.161 | 0.484 | 0.354 | 155.168 |
| BAAI/bge-base-en-v1.5 | fluent | BAAI/bge-reranker-base | 0.286 | 0.169 | 0.444 | 0.351 | 838.434 |
| intfloat/e5-small-v2 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.255 | 0.221 | 0.424 | 0.350 | 93.357 |
| BAAI/bge-small-en-v1.5 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.268 | 0.176 | 0.432 | 0.348 | 103.136 |
| BAAI/bge-small-en-v1.5 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.245 | 0.183 | 0.487 | 0.345 | 148.184 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | BAAI/bge-reranker-base | 0.268 | 0.171 | 0.433 | 0.338 | 601.657 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.245 | 0.153 | 0.396 | 0.319 | 105.073 |
| BAAI/bge-small-en-v1.5 | fluent | BAAI/bge-reranker-base | 0.245 | 0.167 | 0.447 | 0.315 | 826.843 |
| intfloat/e5-small-v2 | moment | none | 0.232 | 0.161 | 0.435 | 0.303 | 8.667 |
| BAAI/bge-small-en-v1.5 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.227 | 0.178 | 0.356 | 0.296 | 92.432 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.204 | 0.148 | 0.389 | 0.288 | 151.640 |
| sentence-transformers/all-MiniLM-L6-v2 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.218 | 0.184 | 0.350 | 0.285 | 92.625 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | BAAI/bge-reranker-base | 0.209 | 0.148 | 0.359 | 0.280 | 844.254 |
| BAAI/bge-small-en-v1.5 | moment | none | 0.218 | 0.145 | 0.378 | 0.278 | 9.018 |
| BAAI/bge-base-en-v1.5 | terse | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.186 | 0.167 | 0.444 | 0.278 | 104.703 |
| sentence-transformers/all-MiniLM-L6-v2 | moment | none | 0.200 | 0.139 | 0.434 | 0.265 | 6.645 |
| intfloat/e5-small-v2 | terse | BAAI/bge-reranker-base | 0.173 | 0.176 | 0.396 | 0.253 | 610.105 |
| BAAI/bge-base-en-v1.5 | terse | BAAI/bge-reranker-base | 0.177 | 0.145 | 0.342 | 0.238 | 624.114 |
| intfloat/e5-small-v2 | fluent | none | 0.168 | 0.165 | 0.374 | 0.232 | 9.711 |
| BAAI/bge-base-en-v1.5 | terse | none | 0.177 | 0.144 | 0.340 | 0.232 | 16.468 |
| intfloat/e5-small-v2 | terse | none | 0.168 | 0.177 | 0.362 | 0.231 | 8.694 |
| sentence-transformers/all-MiniLM-L6-v2 | terse | BAAI/bge-reranker-base | 0.186 | 0.169 | 0.285 | 0.228 | 618.213 |
| BAAI/bge-base-en-v1.5 | fluent | none | 0.182 | 0.140 | 0.281 | 0.226 | 18.121 |
| BAAI/bge-small-en-v1.5 | terse | BAAI/bge-reranker-base | 0.150 | 0.151 | 0.297 | 0.212 | 600.793 |
| BAAI/bge-small-en-v1.5 | fluent | none | 0.150 | 0.141 | 0.322 | 0.209 | 10.036 |
| BAAI/bge-small-en-v1.5 | terse | none | 0.123 | 0.125 | 0.228 | 0.169 | 8.805 |
| sentence-transformers/all-MiniLM-L6-v2 | terse | none | 0.100 | 0.064 | 0.156 | 0.101 | 6.617 |
| sentence-transformers/all-MiniLM-L6-v2 | fluent | none | 0.050 | 0.026 | 0.110 | 0.050 | 7.559 |
| random baseline | - | - | 0.016 | 0.007 | 0.049 | 0.017 | 0.000 |

Chosen on dev: **intfloat/e5-small-v2**, **moment** template, reranker **cross-encoder/ms-marco-MiniLM-L6-v2**.

## Per-query results (intfloat/e5-small-v2 | moment | cross-encoder/ms-marco-MiniLM-L6-v2)

| split | query | n_relevant | p@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| dev | nodded in from a corner | 5 | 0.100 | 0.100 | 0.098 |
| dev | spot kick that the keeper stopped | 15 | 0.100 | 0.167 | 0.078 |
| dev | screamer from way outside the area that went in | 9 | 0.000 | 0.000 | 0.000 |
| dev | cheeky lob over the goalie | 13 | 0.100 | 0.100 | 0.064 |
| dev | bicycle kick attempt | 7 | 0.200 | 0.143 | 0.171 |
| dev | volleyed home after a ball into the box from the wing | 10 | 0.000 | 0.030 | 0.000 |
| dev | fast break goal in the closing minutes | 3 | 0.000 | 0.000 | 0.000 |
| dev | unbelievable miss from a few yards out | 43 | 0.100 | 0.167 | 0.078 |
| dev | shot rattled the crossbar | 39 | 0.300 | 0.333 | 0.252 |
| dev | player failed to convert in the shootout | 15 | 0.300 | 1.000 | 0.469 |
| dev | curled a set piece straight in | 2 | 0.000 | 0.000 | 0.000 |
| dev | late leveller to rescue a draw | 6 | 0.000 | 0.000 | 0.000 |
| dev | last-gasp match winner in added time | 3 | 0.200 | 0.500 | 0.531 |
| dev | followed up and tucked it in after the keeper parried | 11 | 0.000 | 0.000 | 0.000 |
| dev | slotted home after a defence-splitting pass | 13 | 0.000 | 0.024 | 0.000 |
| dev | finish from a pull-back from the byline | 7 | 0.100 | 0.250 | 0.118 |
| dev | flying header | 14 | 0.000 | 0.000 | 0.000 |
| dev | flick with the heel | 5 | 0.000 | 0.000 | 0.000 |
| dev | volley from distance | 106 | 1.000 | 1.000 | 1.000 |
| dev | goalkeeper wins the one against one duel | 37 | 0.600 | 0.333 | 0.461 |
| dev | early goal in the opening minutes | 15 | 0.500 | 1.000 | 0.649 |
| dev | left foot rocket from range into the net | 7 | 0.300 | 1.000 | 0.586 |
| dev | bundled in off the chest or knee | 13 | 0.100 | 0.250 | 0.095 |
| dev | goal that cut the deficit | 13 | 0.000 | 0.045 | 0.000 |
| dev | skied effort nowhere near the goal | 85 | 0.700 | 1.000 | 0.782 |
| dev | rolled into an empty net | 13 | 0.000 | 0.000 | 0.000 |
| dev | converted penalty during normal play | 17 | 0.500 | 1.000 | 0.628 |
| dev | header from a free-kick delivery that went in | 1 | 0.000 | 0.048 | 0.000 |
| dev | goal in the tournament final | 12 | 0.000 | 0.021 | 0.000 |
| dev | shot off target from long distance under pressure | 4 | 0.000 | 0.000 | 0.000 |
| dev | Messi spot kick | 7 | 0.400 | 0.143 | 0.341 |
| test | towering header that hit the back of the net | 28 | 1.000 | 1.000 | 1.000 |
| test | penalty blazed over or wide | 6 | 0.000 | 0.059 | 0.000 |
| test | keeper saves a long-distance drive | 79 | 1.000 | 1.000 | 1.000 |
| test | tap-in from right in front of goal | 33 | 0.000 | 0.031 | 0.000 |
| test | goal to make it all square | 26 | 0.000 | 0.024 | 0.000 |
| test | breakaway goal after winning the ball back | 10 | 0.000 | 0.000 | 0.000 |
| test | half-volley charged down by a defender | 60 | 0.900 | 1.000 | 0.936 |
| test | dead-ball strike kept out by the keeper | 11 | 0.000 | 0.042 | 0.000 |
| test | goal from a ball whipped in from the flank | 35 | 0.000 | 0.000 | 0.000 |
| test | header that hit the frame of the goal | 8 | 0.300 | 1.000 | 0.539 |
| test | goal just before half-time | 5 | 0.000 | 0.000 | 0.000 |
| test | goal that made it a two-goal cushion or more | 56 | 0.000 | 0.043 | 0.000 |
| test | successful kick in the shootout | 26 | 0.600 | 0.250 | 0.462 |
| test | attempt after a corner routine that the keeper held | 29 | 0.500 | 1.000 | 0.633 |
| test | glorious heading chance wasted | 6 | 0.000 | 0.000 | 0.000 |
| test | shot after beating a man with a dribble | 3 | 0.000 | 0.000 | 0.000 |
| test | shot took a nick off a defender and went in | 8 | 0.000 | 0.000 | 0.000 |
| test | volley that the goalkeeper kept out | 55 | 1.000 | 1.000 | 1.000 |
| test | right-footer from just outside the area into the bottom corner | 1 | 0.000 | 0.000 | 0.000 |
| test | goal scored in the extra period | 4 | 0.200 | 1.000 | 0.503 |
| test | Mbappé scoring in the final | 4 | 0.300 | 1.000 | 0.805 |
| test | Richarlison acrobatic finish | 3 | 0.300 | 1.000 | 0.933 |
