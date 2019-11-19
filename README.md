# INCAL
INCAL is an incremental SMT constraint learner


## Installation
You can either clone the repository from GitHub:

    git clone https://github.com/ML-KULeuven/incal.git
    git checkout cleanup
    pip install -e .
    
or install incal via pypi:

    pip install incal
    

You need to install an SMT solver, which you can do using:

    pysmt-install --msat
    
  
## Experiments
To run experiments using autodora, you can specify the experiment-database's location by setting the `DB` environment
variable.
 
### Preset experiments
If you cloned the repository from GitHub, you can use scripts in the ./res/synth folder to quickly generate synthetic
execute:

    python res/synth/small/generate.py

To run experiments you can use:

    python -m incal.run_experiments res/synth/small