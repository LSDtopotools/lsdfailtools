# -*- coding: utf-8 -*-

"""Top-level package for plot_Iverson_lsdtt."""

__author__ = """Boris Gailleton"""
__email__ = 'b.gailleton@sms.ed.ac.uk'
__version__ = '0.1.0'

import sys

if(sys.version[0] == 2):
	from preprocessing import *
	from plot_iverson_lsdtt import PLOTIV
	from sensitivity_analysis import *

else:
	from .preprocessing import *
	from .plot_iverson_lsdtt import PLOTIV
	from .sensitivity_analysis import *

