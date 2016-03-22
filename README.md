# TD-Gammon

Implementation of [TD-Gammon](http://www.bkgm.com/articles/tesauro/tdl.html) in TensorFlow.

Before DeepMind + Atari there was TD-Gammon, an algorithm that combined reinforcement learning and neural networks to play backgammon at an intermediate level with raw features and expert level with hand-engineered features. This is an implementation using raw features: one-hot encoding of each point on the board.

The code also features [eligibility traces](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node87.html#fig:GDTDl) on the gradients which are an elegant way to assign credit to actions made in the past.

## Training

### Cloud Setup

1. Follow the [installation guide](https://fomoro.gitbooks.io/guide/content/installation.html) for Fomoro.
2. Clone the repo: `git clone https://github.com/fomorians/td-gammon.git && cd td-gammon`
3. Create a new model: `fomoro model create`
4. Start training: `fomoro session start`
5. Follow the logs: `fomoro session logs -f`

### Local Setup

1. [Install TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).
2. Clone the repo: `git clone https://github.com/fomorians/td-gammon.git && cd td-gammon`
3. Run training: `python main.py`

# Play

To play against a trained model: `python main.py --play`
