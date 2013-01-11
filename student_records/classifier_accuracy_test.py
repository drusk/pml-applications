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

# Some additional useful functions
import util

def knn_accuracy_tests(training, testing):
    util.print_line_break()
    print "KNN accuracy test:"
    print "k\tAccuracy"
    print "-\t--------"
    
    # Test the accuracy for k values of 3, 4, 5, 6, 7, 8, 9
    for k in range(3, 10):
        accuracy = Knn(training, k=k).classify_all(testing).compute_accuracy()
        print "%d\t%2.5f %%" % (k, 100 * accuracy)
    
def naive_bayes_accuracy_tests(training, testing):
    util.print_line_break()
    print "Naive Bayes accuracy test:"
    
    accuracy = NaiveBayes(training).classify_all(testing).compute_accuracy()
    print "%2.5f %%" % (100 * accuracy)
    
def decision_tree_accuracy_tests(training, testing):
    util.print_line_break()
    print "Decision tree accuracy test:"
    
    # bin grades of 0-3 as low, 4-6 as mid, 7-9 as high
    training.bin("*", [4, 7], bin_names=["low", "mid", "high"])
    testing.bin("*", [4, 7], bin_names=["low", "mid", "high"])
    
    accuracy = DecisionTree(training).classify_all(testing).compute_accuracy()    
    print "%2.5f %%" % (100 * accuracy)

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")

    # Take a 50-50 split
    training, testing = data.split(0.5, using_labels=True)
    
    # Run tests for each classifier to determine the accuracy it can achieve.
    knn_accuracy_tests(training, testing)
    naive_bayes_accuracy_tests(training, testing)
    decision_tree_accuracy_tests(training, testing)

if __name__ == "__main__":
    main()
