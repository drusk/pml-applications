"""
Performs clustering on all three groups of students (successful, probation, 
failed) and then on the merged pass/fail data.  Calculates metrics to 
determine the quality of each clustering.
"""

from pml.api import *

import util

def cluster_3_groups(data):
    print "Success/Probation/Fail clustering:"
    clustered = kmeans(data, k=3)
    purity = clustered.calculate_purity()
    rand_index = clustered.calculate_rand_index()
    
    print "Purity = %.5f, Rand index = %.5f" % (purity, rand_index)

def cluster_2_groups(data):
    print "Pass/Fail clustering:"
    
    # Count successful and probation students as one group (s)
    data.combine_labels(["s", "p"], "s")
    
    clustered = kmeans(data, k=2)
    purity = clustered.calculate_purity()
    rand_index = clustered.calculate_rand_index()
    
    print "Purity = %.5f, Rand index = %.5f" % (purity, rand_index)

def cluster_2_groups_with_pca(data):
    cluster_2_groups(pca(data, 2))

def cluster_3_groups_with_pca(data):
    cluster_3_groups(pca(data, 2))

def main():
    # The original data set.
    data = util.load_data()
    
    # Fill in missing values with the average for that course.
    data.fill_missing_with_feature_means()
    
    cluster_3_groups(data.copy())
    cluster_2_groups(data.copy())
    
    util.print_line_break()
    
    print "Now with PCA:"
    cluster_3_groups_with_pca(data.copy())
    cluster_2_groups_with_pca(data.copy())
    
if __name__ == "__main__":
    main()
