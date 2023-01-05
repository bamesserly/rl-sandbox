Miscellaneous projects while I teach myself RL.

<h3>Setup/Install</h3>

```
conda env create -f rl_conda_env.yml; conda activate rl_conda_env
```
will get you almost all the way there. But this throws an error on the [`keras-rl2`](https://github.com/taylormcnally/keras-rl2) dependence for which I do:

```
git clone https://github.com/wau/keras-rl2.git; cd keras-rl2; python setup.py install; cd ..
```

<details>
  <summary><h3>shower.py</h3> - Q table solution to the beginner RL example</summary>
  
[shower.py](https://github.com/bamesserly/learning-rl/blob/main/shower.py)

My implementation of the common beginner RL example of keeping the temparature
of the shower within the desired range.

Built from scratch with no dependence on external packages!

This script implements the Bellman equation and builds/uses a q-table from
scratch.

Working well at the moment. With a starting temperature near the target,
the table only needs ~10 training episodes to get a perfect score. With a
larger starting temperature range (+/- 20), it needs ~75 training episodes
to cap its score at a max value of 45.

</details>

<details>
  <summary><h3>shower.ipynb</h3> - Notebook of the Q table solution to the shower problem</summary>

[shower.ipynb](https://github.com/bamesserly/learning-rl/blob/main/shower_rl_example.ipynb)

Notebook implementation of the shower problem. Identical q table solution
as in shower.py, except this one breaks the shower environment out in a
openai gym style.

As with `shower.py`, no reliance on other packages, except for numpy. And gym.spaces are used for the action/observation spaces, though they could easily be replaced with numpy arrays or just dicts.
</details>

<h3>Next</h3>

- [x] refactor to clean up and better-resemble the gym-style
  environment-agent-model design (see `shower.ipynb`).
- [x] put it into a notebook (see `shower.ipynb`).
- [x] systematically vary hyper parameters and visualize results.
- [ ] Experiment with other things like rewards.
- [ ] Try this method on another simple game.
- [ ] Can this method be used for multi-agent games?
- [ ] Make it deep!

Next is ... probably to move on to another game. Or make it deep.

<h3>References</h3>

* First place I saw this [shower example](https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment). The deep q-learning here does _not_ work.
* Nice implementation of [Q table solution](https://github.com/bhattacharyya/reach_circle)
