package ensemble

import ensemble.MetaDataExtractor.MetaData
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * Represents a Multinodal Naive Bayes model for classification.
 * Extends Model2 and implements Serializable.
 *
 * @param smoothing  The smoothing factor for feature probabilities.
 * @param metaData   The metadata of the dataset.
 */
class MultinodalNaiveBayesModel(smoothing: Double, var metaData: Seq[MetaData]) extends Model2 with Serializable{
  private var classProbabilities: mutable.Map[Long, Double] = mutable.Map()
  private var featureProbabilities: mutable.Map[Long, mutable.Map[Int, mutable.Map[Double, Double]]] = mutable.Map()

  /**
   * Trains the model using the provided dataset.
   *
   * @param data  The training dataset as an array of Row objects.
   */
  override def train(data: Array[Row]): Unit = {
    val tempData = data.map(row => (row.getLong(0), row.getAs[linalg.Vector](1)))
    val updatedData = UpdateMetadata(tempData).map { dataPoint =>
      Row(dataPoint.label, Vectors.dense(dataPoint.features.toArray))
    }.toArray
    val (classCounts, totalData) = updatedData.par.foldLeft((mutable.Map[Long, Double](), 0.0)) { case ((map, total), row) =>
      val label = row.get(0).asInstanceOf[Double].toLong
      map(label) = map.getOrElse(label, 0.0) + 1.0
      (map, total + 1.0)
    }

    classProbabilities = mutable.Map(classCounts.mapValues(_ / totalData).toSeq: _*)

    val featureProbabilities = updatedData.par.foldLeft(mutable.Map[Long, mutable.Map[Int, mutable.Map[Double, Double]]]()) { case (map, row) =>
      val label = row.get(0).asInstanceOf[Double].toLong
      val features = row.get(1).asInstanceOf[linalg.Vector] match {
        case v: DenseVector => v.toArray.zipWithIndex
        case v: SparseVector => v.toDense.toArray.zipWithIndex
      }

      features.foreach { case (feature, index) =>
        val featureMap = map.getOrElseUpdate(label, mutable.Map[Int, mutable.Map[Double, Double]]())
        val countMap = featureMap.getOrElseUpdate(index, mutable.Map[Double, Double]())
        countMap(feature) = countMap.getOrElse(feature, 0.0) + 1.0
      }

      map
    }

    featureProbabilities.foreach { case (_, featureMap) =>
      featureMap.foreach { case (_, countMap) =>
        val total = countMap.values.sum
        countMap.keys.foreach { key =>
          countMap(key) /= total
        }
      }
    }

    this.featureProbabilities = featureProbabilities
  }

  /**
   * Predicts the labels for the given data points.
   *
   * @param dataPoints  The array of data points to predict, each represented as a tuple of (id, features).
   * @return            The predicted labels as an array of tuples (id, label).
   */
  override def predict(dataPoints: Array[(Long, linalg.Vector)]): Array[(Long, Long)] = {
    dataPoints.map { case (id, features) =>
      val probabilities = classProbabilities.map { case (label, classProb) =>
        val featureProb = features match {
          case v: DenseVector => v.toArray.zipWithIndex
          case v: SparseVector => v.toDense.toArray.zipWithIndex
        }
        featureProb.map { case (feature, index) =>
          featureProbabilities(label).getOrElse(index, mutable.Map()).getOrElse(feature, smoothing)
        }.sum + math.log(classProb)
      }
      id -> probabilities.max.toLong
    }
  }

  /**
   * Converts real-valued features to bucketed features based on the metadata.
   *
   * @param dataPointsIterator  An iterator of data points, each represented as a tuple of (id, features).
   * @param metaDataSeq         The metadata of the dataset.
   * @return                    An iterator of DataPoints with bucketed features.
   */
  private def UpdateMetadata(dataPointsIterator: Array[(Long, linalg.Vector)]): List[DataPoint] = {
    val data = dataPointsIterator.toList
    val newMetaData = mutable.ArrayBuffer.fill(metaData.length)(mutable.Set.empty[Double])
    val newDataPoints: ListBuffer[DataPoint] = ListBuffer.empty[DataPoint]
    data.foreach(value => {
      val features = value._2.toArray
      val newFeatures = Array.range(0,features.size).map(i => {
        val feature = value._2(i)
        val valueMetaData = metaData(i)
        if(!valueMetaData.isCategorical)
        {
          val bucketSize = valueMetaData.numDistinctValues/1000
          val divider = (valueMetaData.max - valueMetaData.min)/bucketSize
          val bucket = math.floor(feature/divider).toInt
          newMetaData(i) += bucket
          bucket.toDouble
        }
        else {
          newMetaData(i) += feature
          feature
        }

      })
      newDataPoints += DataPoint(label = value._1, features = linalg.Vectors.dense(newFeatures))
    })
    metaData = newMetaData.map(value =>
      MetaData(isCategorical = true, isOrdinal = true, numDistinctValues = value.size, max = value.max, min = value.min, values = value.toSeq.sorted)
    )
    newDataPoints.toList
  }
}