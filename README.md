# TD-Gammon

Implementation of [TD-Gammon](http://www.bkgm.com/articles/tesauro/tdl.html) in TensorFlow.

Before DeepMind tackled playing Atari games or built AlphaGo there was TD-Gammon, the first algorithm to reach an expert level of play in backgammon. Gerald Tesauro published his paper in 1992 describing TD-Gammon as a neural network trained with reinforcement learning. It is referenced in both Atari and AlphaGo research papers and helped set the groundwork for many of the advancements made in the last few years.

The code features [eligibility traces](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node87.html#fig:GDTDl) on the gradients which are an elegant way to assign credit to actions made in the past.

## Training

1. [Install TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).
2. Clone the repo: `git clone https://github.com/fomorians/td-gammon.git && cd td-gammon`
3. Run training: `python main.py`

## Play

To play against a trained model: `python main.py --play --restore`

## Things to try

- Compare with and without eligibility traces by replacing the trace with the unmodified gradient.
- Try different activation functions on the hidden layer.
- Expand the board representation. Currently it uses the "compact" representation from the paper. A full board representation should remove some ambiguity between board states.
- Increase the number of turns the agent will look at before making a move. The paper used a 2-ply and 3-ply search while this implementation only uses 1-ply.
