"""
Some functions useful for a variety of analysis scripts.
"""
# Python standard library imports
import sys
import inspect

from pml.api import load

def print_line_break():
    print "*" * 50
    
def load_data():
    """
    Loads data from the file whose name/path is passed in when calling 
    the script.
    """
    if len(sys.argv) != 2:
        # This stack inspection is to find the name of the script that was 
        # originally ran, as opposed to this module.
        print "Usage: python %s <file_path>" % inspect.stack()[-1][1]
        sys.exit(1)
        
    filename = sys.argv[1]
    
    return load(filename)