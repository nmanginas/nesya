## Introduction
This is code necessary to reproduce the experiments for NeSyA: Neurosymbolic Automata to appear
in IJCAI 2025.

Much of the work is based on the [deepfa](https://github.com/nmanginas/deepfa), which does the 
heavy lifting and inference. The repository is also cloned here for ease of installation and for 
reproducibility but for detailed instructions to create one's own applications please refer
to the deepfa repo above. For a baseline we use [deepstochlog](https://github.com/ML-KULeuven/deepstochlog), the
authors of which we give thanks to. We include the code here also for version fixing and experimental reproducibility.


## Usage

Install the package with poetry using:
```poetry install```

All results have been run and are available in the results directory.

You can see the results for the synthetic benchmark:
```python nesya/driving/vizualize.py```

and for CAVIAR with:
```python nesya/caviar/results.py```


To regenerate the results for the synthetic benchmark use:
```python nesya/driving/train.py``` and ```python nesya/driving/scalability/scalability.py```

Note that this will take a very long amount of time due to the unscalable baselines. Also for the scalability 
experiments DeepProbLog is skipped but there is a flag in the program to re-enable it. It will consume all of 
the memory in a 128GB machine quickly (in the second pattern).

To regenerate the results for the caviar benchmark use:
```python nesya/caviar/nesy.py```, ```python nesya/caviar/pure_neural.py``` and
```python nesya/caviar/transformer.py``` for NeSyA, the CNN-LSTM and the CNN-Transformer
respectively.

