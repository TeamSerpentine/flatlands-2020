# The Team: Serpentrain

This is the finalized repo from Serpentrain, the 2020 [SerpentineAI](serpertine.ai) team that competed in the
[2020 flatland challenge](https://flatland.aicrowd.com/). The technical report of this and other competitions joined by
teams from SerpentineAI can be found [online](https://serpentine.ai/publications/).

# Run the code

The competition uses [conda](https://conda.io) as package manager. After installing conda (or miniconda) run the
following commands in the repo root (~/../flatland)
to download and install all requirements:

```bash
conda env create
conda activate flatland-rl
```

## DQN Agent

### Training

Training our agent takes 3 gpus, as it was what we had access to during the competition. To train our agent run:

```bash
PYTHONPATH=. python serpentrain/reinforcement_learning/distributed/main_distributed.py
```

Warning this might freeze up your system as it is resource heavy.

To see the various training options you can run:

```bash
PYTHONPATH=. python serpentrain/reinforcement_learning/distributed/main_distributed.py -h
```

### Running

Adjust the run.py file.

```python
RENDER = True  # Whether to render the game 
USE_GPU = True  # If you have a GPU 
DQN_MODEL = True
CHECKPOINT_PATH = "path/to/checkpoint.pt"  # E.G. './checkpoints/submission/snapshot-20201104-2201-epoch-1.pt'
```

Then run:

```bash
bash local_run.sh
```

## Rule Based Agent

Adjust the run.py file.

```python
RENDER = True  # Whether to render the game
USE_GPU = False  # Not necessary
DQN_MODEL = False
CHECKPOINT_PATH = ""  # Not necessary
```

Then run:

```bash
bash local_run.sh
```

# Acknowledgements

## SerpentineAI

SerpentineAI is a [student team](https://www.tue.nl/en/our-university/student-teams/) from the
[Technical University of Eindhoven](https://www.tue.nl/en). During the competition computational resources provided by
[VBTI](https://vbti.nl/) to SerpentineAI were used during training.
