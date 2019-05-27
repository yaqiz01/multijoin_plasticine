# Multijoin on Plasticine
This repository contains the peformance evaluation of multiway hash join algorithms on the Plastcine accelerator.
Two types of joins are considered:
1. *Self Join*: Joining 3 copies of a large table twice, where the table does not fit on-chip, e.g. the join of a friend relation table produces a table for friend of friend of friend relations.
2. *Star Join*: Joining of two small tables with K records with a large middle table with N records, where K fits on-chip but N does not.
1. `model.py`: performance model of multiway hash join on Plasticine.
   * `def run_algo(**kws)`: entry function of the model. `kws` contains a set of problem, algorithm, and architectural parameters. For all possible parameters and their defaults, please take a look at the `init_param` function.
   * Possible values for the `algo` key:
      * `join2`: cascaded binary self join
      * `join3`: 3-way self join
      * `starjoin2`: cascaded binary star join
      * `starjoin3`: 3-way star join
2. `cpu.py`: cpu comparision with MySQL and PostgreSQL.
3. `plot`: ploting script to generate figures in the paper.
