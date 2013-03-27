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

import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.io.File;
import java.io.IOException;
import java.util.regex.Pattern;

public class Utils {

  private Utils() {}

  public static Matrix loadOccurrences(File occurrences, int numRows, int numColumns) throws IOException {

    Matrix A = new SparseRowMatrix(numRows, numColumns);

    Pattern splitter = Pattern.compile("\t");
    Pattern splitter2 = Pattern.compile(":");
    for (String line : new FileLineIterable(occurrences)) {
      String[] parts = splitter.split(line);

      if (parts.length > 1) {
        int entityIndex = Integer.parseInt(parts[0]);

        for (int index = 1; index < parts.length; index++ ) {
          String[] tokens = splitter2.split(parts[index]);
          int patternIndex = Integer.parseInt(tokens[0]);
          double value = Double.parseDouble(tokens[1]);

          A.setQuick(patternIndex - 1, entityIndex - 1, value);
        }
      }
    }
    return A;
  }

  public static OpenIntObjectHashMap<String> loadLabels(File patternsFile) throws IOException {
    OpenIntObjectHashMap labels = new OpenIntObjectHashMap();
    Pattern splitter = Pattern.compile("\t");
    for (String line : new FileLineIterable(patternsFile)) {
      String[] parts = splitter.split(line);
      labels.put(Integer.parseInt(parts[0]) - 1, parts[1]);
    }
    return labels;
  }

}
