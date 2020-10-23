# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:51:31 2020

@author: Jure
"""

class BaseExpressionGenerator:
    def __init__ (self):
        self.generator_type = "base"
    
    def generate_one (self):
        return "x"
    
