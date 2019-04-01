# INCAL
INCAL is an incremental SMT constraint learner


## Installing LP Learning
Obtain the LP learning code
- `git clone https://github.com/samuelkolb/incal.git`
- `cd incal`
- `git checkout LP`
- `virtualenv -p python3 env` *[env-only]*

Add gurobipy to Python installation
- Navigate to gurobi installation (e.g. `cd /Library/gurobi810/mac64/`)
- `python setup.py install`

Install Libraries
- `pip install pebble pywmi sklearn`

## Running LP Learning
- `source env/bin/activate` *[env-only]*
- `cd smtlearn`
- `python lplearing.py syn <v> <h> <method>`
