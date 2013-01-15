"""
Examines the principal components of the data set.
"""

from pml.api import *

import util

def examine_principal_components(dataset):
    print "Examining principal components for the data set."
    print "Percent variance per principal component:"
    print get_pct_variance_per_principal_component(dataset)
    plot_pct_variance_per_principal_component(dataset)
    
    print "Recommended number of principal components for 95% variance:"
    print recommend_num_components(dataset, min_pct_variance=0.95)

def pca_find_important_features(dataset):
    # The weight matrix that is used to transform the original data 
    # to the reduced data can be used to see which features are most 
    # important.  The features with the largest magnitude weight have 
    # the largest impact on the reduced data.
    
    util.print_line_break()
    print "First principal component impacts (absolute value of weight):"
    print pca(dataset, 2).get_first_component_impacts()

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")
    
    examine_principal_components(data)
    
    pca_find_important_features(data)

if __name__ == "__main__":
    main()
    