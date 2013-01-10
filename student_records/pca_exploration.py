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

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    # Count successful and probation students as one group (s)
    # Comment this out to try and distinguish all 3 groups (s, p, f)
    data.combine_labels(["s", "p"], "s")
    
    examine_principal_components(data)

if __name__ == "__main__":
    main()
    