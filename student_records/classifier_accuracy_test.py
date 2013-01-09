"""
Determines the accuracy of KNN, Naive Bayes and Decision Tree classifiers
at predicting student performance.

The features of the data set are course grades with values on a scale from 
0 to 9.  Each student is labelled as either having succeeded (s), been on 
probation (s) but still passed, or failed (f).
"""

# Import the PML interactive api so that we can program like we were at 
# the PML shell.
from pml.api import *

def knn_accuracy_tests(training, testing):
    print_line_break()
    print "KNN accuracy test:"
    print "k\tAccuracy (%)"
    print "-\t--------"
    
    # Test the accuracy for k values of 3, 4, 5
    for k in range(3, 6):
        accuracy = Knn(training, k=k).classify_all(testing).compute_accuracy()
        print "%d\t%2.5f" % (k, 100 * accuracy)
    
def naive_bayes_accuracy_tests(training, testing):
    print_line_break()
    print "Naive Bayes accuracy test:"
    
    accuracy = NaiveBayes(training).classify_all(testing).compute_accuracy()
    print "%2.5f" % (100 * accuracy)
    
def decision_tree_accuracy_tests(training, testing):
    print_line_break()
    print "Decision tree accuracy test:"
    print "<Decision Tree results>"

def main():
    # The original data set.
    data = load_data()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")
    
    # Randomly split 70% of samples into the training set, and use the 
    # remaining 30% for a testing data set.
    training, testing = data.split(0.7, random=True)
    
    # Run tests for each classifier to determine the accuracy it can achieve.
    knn_accuracy_tests(training, testing)
    naive_bayes_accuracy_tests(training, testing)
    decision_tree_accuracy_tests(training, testing)

def print_line_break():
    print "*" * 50
    
def load_data():
    """
    Loads data from the file whose name/path is passed in when calling 
    the script.
    """
    # Python standard library imports
    import sys
    import os.path

    if len(sys.argv) != 2:
        print "Usage: python %s <file_path>" % os.path.basename(__file__)
        sys.exit(1)
        
    filename = sys.argv[1]
    
    return load(filename)

if __name__ == "__main__":
    main()
