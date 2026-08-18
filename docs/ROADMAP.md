# Roadmap

## 8C — current
Production package refactor and repository stabilization.

## Dataset
Build controlled geometry-interpolation bank generator and a resumable,
sharded writer for the 4,000 x 32 x 512 label dataset.

## q05 scorer
Train a physics-informed NN to predict `log(q05GG)` from deployable
environment/W/z/analytic features. Compare regression and bank-ranking
metrics.

## High-q05 enrichment
Use the scorer for multi-start bit-flip / multi-bit search, verify finalists
with 64k MC, append verified high-q05 candidates, and retrain scorer v2.

## Actor
Generate verified top-k RIS teachers per `(environment,W)` and train Actor.
Final evaluation is regret against 64k-MC verified q05GG.
