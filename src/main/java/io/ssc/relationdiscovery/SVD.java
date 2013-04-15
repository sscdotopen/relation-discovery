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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Ints;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.decomposer.EigenStatus;
import org.apache.mahout.math.decomposer.SimpleEigenVerifier;
import org.apache.mahout.math.decomposer.lanczos.LanczosSolver;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class SVD {

  private final Matrix A;
  private final LanczosState lanczosState;
  private int rank;
  private List<WeightedVector> singularVectors = Lists.newArrayList();

  private static final int OVERSHOOT = 10;
  private static final double MAX_ERROR = 0.05;
  private static final double MIN_EIGENVALUE = 0;

  private static final Logger log = LoggerFactory.getLogger(SVD.class);

  public SVD(Matrix A, int rank) throws IOException {
    this.A = A;
    this.rank = rank;

    Vector initialVector = new DenseVector(A.numCols());
    initialVector.assign(1.0 / Math.sqrt(A.numCols()));

    lanczosState = new LanczosState(A, rank + OVERSHOOT, initialVector);
  }

  public void compute() {

    new LanczosSolver().solve(lanczosState, rank + OVERSHOOT, false);

    Matrix singularVectorCandidates = new DenseMatrix(rank + OVERSHOOT, A.numCols());
    for (int n = 0; n < rank + OVERSHOOT; n++) {
      singularVectorCandidates.assignRow(n, lanczosState.getRightSingularVector(n));
    }

    findSingularVectors(singularVectorCandidates);
  }

  private void findSingularVectors(Iterable<MatrixSlice> singularVectorCandidates) {

    Map<MatrixSlice, EigenStatus> eigenMetaData = Maps.newHashMap();

    SimpleEigenVerifier verifier = new SimpleEigenVerifier();
    for (MatrixSlice slice : singularVectorCandidates) {
      EigenStatus status = verifier.verify(A, slice.vector());
      eigenMetaData.put(slice, status);
    }

    List<Map.Entry<MatrixSlice, EigenStatus>> prunedEigenMeta = Lists.newArrayList();

    for (Map.Entry<MatrixSlice, EigenStatus> entry : eigenMetaData.entrySet()) {
      if (Math.abs(1 - entry.getValue().getCosAngle()) < MAX_ERROR
          && entry.getValue().getEigenValue() > MIN_EIGENVALUE) {
        prunedEigenMeta.add(entry);
      }
    }

    Collections.sort(prunedEigenMeta, new Comparator<Map.Entry<MatrixSlice, EigenStatus>>() {
      @Override
      public int compare(Map.Entry<MatrixSlice, EigenStatus> e1, Map.Entry<MatrixSlice, EigenStatus> e2) {
        return Ints.compare(e1.getKey().index(), e2.getKey().index());
      }
    });

    int limit = prunedEigenMeta.size() > rank ? rank : prunedEigenMeta.size();
    for (int n = 0; n < limit; n++) {
      Map.Entry<MatrixSlice, EigenStatus> entry = prunedEigenMeta.get(n);
      double eigenvalue = Math.sqrt(entry.getValue().getEigenValue());
      log.info("Eigenvalue {}: {}", n, eigenvalue);
      singularVectors.add(new WeightedVector(entry.getKey().vector(), eigenvalue, n));
    }

  }

  public Matrix projectRowsOntoFeatureSpace() {

    SparseRowMatrix projection = new SparseRowMatrix(A.numRows(), rank);

    for (int patternIndex = 0; patternIndex < A.numRows(); patternIndex++) {

      Vector patternOccurrences = A.viewRow(patternIndex);

      for (int r = 0; r < rank; r++) {
        WeightedVector singularVector = singularVectors.get(r);
        double weight = singularVector.getWeight() * patternOccurrences.dot(singularVector);
        projection.setQuick(patternIndex, r, weight);
      }
    }
    return projection;
  }

}
