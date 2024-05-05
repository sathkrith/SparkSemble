package ensemble

import scala.collection.mutable
import scala.collection.parallel.mutable.ParArray
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row

class BucketedKNNModel(override val k: Int, numBuckets: Int) extends KNNWithoutRDD(k) {
  val buckets: Array[mutable.ArrayBuffer[(Long, Vector)]] = Array.fill(numBuckets)(mutable.ArrayBuffer.empty)

  override def train(dataPassed: Array[Row]) {
    data = dataPassed.map { case Row(label: Long, features: Vector) =>
      (label, features)
    }

    // Bucketize the training data
    data.foreach { case (label, features) =>
      val bucketId = bucketize(features)
      buckets(bucketId) += ((label, features))
    }
  }

  // Function to assign a point to a bucket based on its feature values
  def bucketize(features: Vector): Int = {
    // Compute a hash of the feature values
    val hash = features.toArray.map(_.hashCode()).sum

    // Use the hash value to determine the bucket
    math.abs(hash) % numBuckets
  }

  override def predict(test_features: Array[(Long, Vector)]): Array[(Long, Long)] = {
    implicit val ordering: Ordering[(Long, Double)] = Ordering.by(_._2)

    val predictions = test_features.par.map { case (index, feature) =>
      // Determine the bucket for the test point
      val bucketId = bucketize(feature)

      // Find the k-nearest neighbors within the same bucket
      val neighbors = buckets(bucketId).map(train_features => (train_features._1, approxEuclidean(train_features._2, feature)))
        .sorted.take(k)

      // Get the labels of the k-nearest neighbors
      val pred = neighbors.map(_._1)

      // Use majority voting to determine the predicted label
      val predictedLabel = pred.groupBy(identity).mapValues(_.size).maxBy(_._2)._1

      (index, predictedLabel)
    }
    predictions.toArray
  }
}