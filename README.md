# INCAL
INCAL is an incremental SMT constraint learner


## Installation
You can either clone the repository from GitHub:

    git clone https://github.com/ML-KULeuven/incal.git
    git checkout cleanup
    pip install -e .
    
or install incal via pypi:

    pip install incal
    

Depending on the solver you may need to install an SMT solver, which you can do using:

    pysmt-install --msat
    
Or Gurobi, which you need to download and subsequently install into your python environment.
