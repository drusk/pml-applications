"""
Check the accuracy of the Knn classifier using PCA with each possible number 
of principal components.
"""

from pml.api import *

import util

def get_knn_accuracy(data):
    training, testing = data.split(0.7)
    return Knn(training, k=5).classify_all(testing).compute_accuracy()

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")
    
    util.print_line_break()
    print "Without PCA: %.5f" % get_knn_accuracy(data)
    
    util.print_line_break()
    print "With PCA:"
    print "\t".join(["PCs", "Accuracy"])
    for num_components in range(1, data.num_features()):
        accuracy = get_knn_accuracy(pca(data, num_components))
        print "%d\t%.5f" % (num_components, accuracy)

if __name__ == "__main__":
    main()
