"""
Compares classification results with and without the use of PCA.
"""

from pml.api import *

import util

def compare_knn(training, testing, pca_training, pca_testing):
    util.print_line_break()
    print "Comparing KNN accuracy with and without PCA"
    
    print "Without PCA:"
    print Knn(training, k=5).classify_all(testing).compute_accuracy()
    
    print "With PCA"
    print Knn(pca_training, k=5).classify_all(pca_testing).compute_accuracy()

def compare_naive_bayes(training, testing, pca_training, pca_testing):
    util.print_line_break()
    print "Comparing Naive Bayes accuracy with and without PCA"
    
    print "Without PCA:"
    print NaiveBayes(training).classify_all(testing).compute_accuracy()
    
    print "With PCA"
    print NaiveBayes(pca_training).classify_all(pca_testing).compute_accuracy()

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")
    
    num_components = recommend_num_components(data, min_pct_variance=0.95)
    pca_data = pca(data, num_components)
    
    training, testing = data.split(0.7)
    pca_training, pca_testing = pca_data.split(0.7)
    
    compare_knn(training, testing, pca_training, pca_testing)
    compare_naive_bayes(training, testing, pca_training, pca_testing)

if __name__ == "__main__":
    main()
