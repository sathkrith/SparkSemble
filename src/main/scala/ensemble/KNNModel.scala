package ensemble

import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.parallel.mutable.ParArray

case class KNNWithoutRDD(k: Int) extends Model2{
    var data:Array[(Long, Vector)] = null
    def approxEuclidean(v1: Vector, v2: Vector): Double = {
        v1.toArray.zip(v2.toArray).map { case (x, y) => math.abs(x - y) }.sum
    }

    def train(dataPassesed: Array[Row]){
        data = dataPassesed.map { case Row(label: Long, features: Vector) =>
            (label, features)
        }
    }
    // Define a function to compute the Euclidean distance
    def euclideanDistance(v1: Vector, v2: Vector): Double = {
        math.sqrt(v1.toArray.zip(v2.toArray).map { case (x, y) => math.pow(x - y, 2) }.sum)
    }

    def predictPoint(test_feature: Vector): Long = {
        implicit val ordering: Ordering[(Long, Double)] = Ordering.by(_._2)


        val neighbors = data.map(train_features => (train_features._1, approxEuclidean(train_features._2, test_feature))).sorted.take(k)

        // Get the labels of the k-nearest neighbors
        val pred = neighbors.map(_._1)

        // Use majority voting to determine the predicted label
        val predictedLabel = pred.groupBy(identity).mapValues(_.size).maxBy(_._2)._1

        predictedLabel
    }

    def predict(test_features: Array[(Long, Vector)]): Array[(Long, Long)] = {
        implicit val ordering: Ordering[(Long, Double)] = Ordering.by(_._2)

        val predictions = test_features.par.map { case (index, feature) =>
            // Find the k-nearest neighbors
            val neighbors = data.map(train_features => (train_features._1, approxEuclidean(train_features._2, feature)))
              .sorted.take(k)

            // Get the labels of the k-nearest neighbors
            val pred = neighbors.map(_._1)

            // Use majority voting to determine the predicted label
            val predictedLabel = pred.groupBy(identity).mapValues(_.length).maxBy(_._2)._1

            (index, predictedLabel)
        }
        predictions.toArray
    }
}

case class KNNModel(k: Int, trainingData: RDD[(Long, Vector)]) extends model{
    def approxEuclidean(v1: Vector, v2: Vector): Double = {
        v1.toArray.zip(v2.toArray).map { case (x, y) => math.abs(x - y) }.sum
    }

    // Define a function to compute the Euclidean distance
    def euclideanDistance(v1: Vector, v2: Vector): Double = {
        math.sqrt(v1.toArray.zip(v2.toArray).map { case (x, y) => math.pow(x - y, 2) }.sum)
    }

    def predict(test_features: Array[(Long, Vector)]): ParArray[(Long, Long)] = {
        implicit val ordering: Ordering[(Long, Double)] = Ordering.by(_._2)

        val predictions = test_features.par.map{case (index, feature) => 
            // Find the k-nearest neighbors
            val neighbors = trainingData.mapValues(train_features => approxEuclidean(train_features, feature))
            .takeOrdered(k)
            
            // Get the labels of the k-nearest neighbors
            val pred = neighbors.map(_._1)

            // Use majority voting to determine the predicted label
            val predictedLabel = pred.groupBy(identity).mapValues(_.size).maxBy(_._2)._1

            (index, predictedLabel)
        }
        predictions

    }
}