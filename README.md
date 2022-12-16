Bunch of misc projects while I teach myself RL.

### Shower.py
My implementation of the common beginner RL example of keeping the temparature
of the shower within the desired range.

This script implements the Bellman equation and builds/uses a q-table from
scratch.

Working well at the moment. With a starting temperature near the target,
the table only needs ~10 training episodes to get a perfect score. With a
larger starting temperature range (+/- 20), it needs ~75 training episodes
to cap its score at a max value of 45.

So what's next?
* refactor to clean up and better-resemble the gym-style
  environment-agent-model design.
* put it into a notebook.
* systematically vary hyper parameters and visualize results.
* Experiment with other things like rewards.
* Try this method on another simple game.
* Can this method be used for multi-agent games?
* Make it deep!

Of these, I think priorities should be to put it into a notebook and clean it
up so it's more gym-style, more notebook-y, and easier to follow. Also: begin
visualizing hyperparameter variation/optimization.
