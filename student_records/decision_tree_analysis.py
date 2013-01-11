"""
Examines decision tree structure.
"""

from pml.api import *

import util

def bin_test(data, boundaries, bin_names):
    data = data.copy()
    data.bin("*", boundaries, bin_names)
    
    training, testing = data.split(0.5, using_labels=True)
    accuracy = DecisionTree(training).classify_all(testing).compute_accuracy()
    print "%s -> %.5f" % (boundaries, accuracy)

def binning_exploration(data):
    bin_test(data, [4, 7], ["low", "mid", "high"])
    bin_test(data, [3, 6], ["low", "mid", "high"])
    bin_test(data, [3, 7], ["low", "mid", "high"])
    bin_test(data, [2, 6], ["low", "mid", "high"])
    bin_test(data, [4], ["low", "high"])
    bin_test(data, [3], ["low", "high"])
    bin_test(data, [2, 6, 8], ["bad", "fair", "good", "excellent"])
    bin_test(data, [3, 6, 8], ["bad", "fair", "good", "excellent"])
    bin_test(data, [3, 6, 7], ["bad", "fair", "good", "excellent"])
    bin_test(data, [3, 6, 9], ["bad", "fair", "good", "excellent"])
    bin_test(data, [3, 7, 9], ["bad", "fair", "good", "excellent"])

def plot_tests(data):
    data.bin("*", [3, 6, 9], bin_names=["bad", "fair", "good", "excellent"])
    training, _ = data.split(0.5, using_labels=True)
    
    tree = DecisionTree(training)
    
    print "Tree with half the data for training"
    tree.plot()
    
    print "Tree with all the data for training"
    tree2 = DecisionTree(data)
    tree2.plot()

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")

    binning_exploration(data)
    plot_tests(data)

if __name__ == "__main__":
    main()
