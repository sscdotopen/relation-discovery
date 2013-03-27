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

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.lanczos.LanczosSolver;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;


import java.io.IOException;

public class SVD {

  private final Matrix A;
  private final LanczosState lanczosState;
  private int rank;

  public SVD(Matrix A, int rank) throws IOException {
    this.A = A;
    this.rank = rank;

    Vector initialVector = new DenseVector(A.numCols());
    initialVector.assign(1.0 / Math.sqrt(A.numCols()));

    lanczosState = new LanczosState(A, rank, initialVector);
  }

  public void compute() {
    new LanczosSolver().solve(lanczosState, rank, false);

    //TODO compute a few more eigenvectors and verify them
  }


  public Matrix projectRowsOntoFeatureSpace() {

    SparseRowMatrix projection = new SparseRowMatrix(A.numCols(), rank);

    for (int patternIndex = 0; patternIndex < A.numCols(); patternIndex++) {

      Vector patternOccurrences = A.viewRow(patternIndex);

      for (int r = 0; r < rank; r++) {
        double weight =
            lanczosState.getSingularValue(r) * patternOccurrences.dot(lanczosState.getRightSingularVector(r));
        projection.setQuick(patternIndex, r, weight);
      }

    }
    return projection;
  }

}
