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

import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.FixedSizeSamplingIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class KMeans {

  private final int k;
  private final Matrix A;
  private final DistanceMeasure distanceMeasure;

  private Vector[] centroids;

  private static final Logger log = LoggerFactory.getLogger(KMeans.class);

  public KMeans(Matrix A, int k, DistanceMeasure distanceMeasure) {
    this.A = A;
    this.k = k;
    this.distanceMeasure = distanceMeasure;

    centroids = new Vector[k];

    log.info("Picking {} initial centroids", k);
    FixedSizeSamplingIterator<MatrixSlice> sampler = new FixedSizeSamplingIterator<MatrixSlice>(k, A.iterator());
    int index = 0;
    while (sampler.hasNext()) {
      centroids[index++] = sampler.next().vector();
    }
  }

  public void run(int numIterations) {

    int iteration = 0;
    while (iteration++ < numIterations) {
      log.info("Running Iteration {}", iteration);
      singleIteration();
    }
  }

  public void printClosestPoints(int centroidIndex, int howMany, OpenIntObjectHashMap<String> patterns) {

    PriorityQueue<PatternWithDistance> queue = new PriorityQueue<PatternWithDistance>(howMany) {

      @Override
      protected boolean lessThan(PatternWithDistance a, PatternWithDistance b) {
        return a.distance < b.distance;
      }
    };

    Vector centroid = centroids[centroidIndex];

    for (MatrixSlice rowSlice : A) {
      Vector row = rowSlice.vector();
      double d = distanceMeasure.distance(centroid, row);
      queue.insertWithOverflow(new PatternWithDistance(d, patterns.get(rowSlice.index())));
    }

    while (queue.size() > 0) {
      System.out.println("\t" + queue.pop());
    }

  }

  private static class PatternWithDistance {

    private double distance;
    private String name;

    PatternWithDistance(double distance, String name) {
      this.distance = distance;
      this.name = name;
    }

    @Override
    public String toString() {
      return name + "(" + distance + ")";
    }
  }


  //TODO might be vulnerable to overflow for large datasets
  private void singleIteration() {

    int[] pointsPerCentroid = new int[k];
    Vector[] nextCentroids = new Vector[k];
    for (int n = 0; n < k; n++) {
      nextCentroids[n] = new DenseVector(A.numCols());
    }

    for (MatrixSlice rowSlice : A) {

      Vector row = rowSlice.vector();

      int nearestCentroid = 0;
      double closestDistance = Double.MAX_VALUE;

      for (int n = 0; n < k; n++) {
        double d = distanceMeasure.distance(centroids[n], row);
        if (d < closestDistance) {
          closestDistance = d;
          nearestCentroid = n;
        }
      }

      nextCentroids[nearestCentroid].assign(row, Functions.PLUS);
      pointsPerCentroid[nearestCentroid]++;
    }

    for (int n = 0; n < k; n++) {
      if (pointsPerCentroid[n] != 0) {
        nextCentroids[n].assign(Functions.DIV, pointsPerCentroid[n]);
      }
    }

    centroids = nextCentroids;
  }

}
