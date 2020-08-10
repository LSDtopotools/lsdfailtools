lsdfailtools
==============

Set of python-c++ tools to use landslide model from Iverson (2000). It provides an easy interface to (i) run single simulations, (ii) run MC simulations from ranges of parameters and (iii) access all the output simplified or as time-depths series. 


Installation
------------

**On Unix (Linux, OS X)**
 
 - You need gcc 5+ to install it
 - `pip install pybind11 numpy pandas`
 - (or `conda install -c conda-forge pybind11 numpy pandas`)
 - clone this repository
 - `python setup.py bdist_wheel`
 - `pip install dist/XXX.whl` <- XXX is the name of the wheel generated by the python line!

**On Windows (Requires Visual Studio 2015)**
 - Same steps but requires Visual Studio <=2015

Quick Start
-----------

The `examples` folder has quick example on running a single analysis or the MC simulation and exploring the results.


Authors
-------

Simon M. Mudd - University of Edinburgh
Boris Gailleton - University of Edinburgh - GFZ Potsdam
Guillaume ""Will"" Goodwin - University of Edinburgh - A University in Italy (I cannot remember which one sorry)