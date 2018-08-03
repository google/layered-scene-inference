# Installation Instructions

The code requires a working implementation of TensorFlow (tested on 1.4.0). To install the additional dependencies (we recommended doing this in a virtualenv), run

```
virtualenv lsi_venv
source lsi_venv/bin/activate
pip install -U pip
deactivate
source lsi_venv/bin/activate
pip install -r docs/requirements.txt
```
