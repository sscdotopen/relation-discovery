/*
 * Copyright (C) 2013 Sebastian Schelter
 *
 * relation-discovery is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * relation-discovery is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with relation-discovery; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 * USA
 */

package io.ssc.relationdiscovery;

import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.io.File;
import java.io.IOException;

public class Runner {

  public static void main(String[] args) throws IOException {

    File labelsFile = new File("/home/ssc/Desktop/alan/R30 Tupel/feature_dict/part-r-00000");
    File occurrencesFile = new File("/home/ssc/Desktop/alan/R30 Tupel/tupleids/occurrences.tsv");

    // number of entity pairs in the data
    int numEntityPairs = 7853;
    // number of patterns in the data
    int numPatterns = 58702;

    // desired rank for dimension reduction
    int rank = 50;

    // distance measure for clustering
    DistanceMeasure distanceMeasure = new CosineDistanceMeasure();

    // number of clusters (k of k-Means)
    int numClusters = 10;
    // maximum number of iterations to run
    int maxIterations = 100;
    // number of points to print per cluster
    int numClosestPointsPerCluster = 20;



    long start = System.currentTimeMillis();

    OpenIntObjectHashMap<String> labels = Utils.loadLabels(labelsFile);

    Matrix A = Utils.loadOccurrences(occurrencesFile, numPatterns, numEntityPairs);

    SVD svd = new SVD(A, rank);
    svd.compute();
    Matrix P = svd.projectRowsOntoFeatureSpace();

    KMeans kMeans = new KMeans(P, numClusters, distanceMeasure);

    kMeans.run(maxIterations);

    for (int n = 0; n < numClusters; n++) {
      System.out.println("-----" + n + "------");
      kMeans.printClosestPoints(n, numClosestPointsPerCluster, labels);
      System.out.println("\n");
    }

    System.out.println("Computation took " + (System.currentTimeMillis() - start) + "ms");
  }



}
