package ensemble

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import scala.util.Random

object KnnEnsemble {
  def main(args: Array[String]): Unit = {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 5) {
      logger.error("Usage:\nensemble.KnnEnsemble <train data dir> <test data dir> numModels K fraction")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("Parallel KNN Ensemble")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    val numModels = args(2).toInt
    val k = args(3).toInt
    val fraction = args(4).toDouble

    val schema = new StructType()
                .add(StructField("label", LongType, false))
                .add(StructField("features", VectorType, false))

    val train_rawdata = spark.sparkContext.textFile(args(0))


    val train_data = spark.createDataFrame(data_process(train_rawdata), schema)
    sc.broadcast(train_data)

    val models = (0 until numModels).par.map { _ =>
      val bootstrapData = train_data.sample(withReplacement = true, fraction = 1.0, seed = Random.nextInt())

      val model = trainKNNModel(bootstrapData, k = k)
      model
    }.toArray

    val test_rawdata = spark.sparkContext.textFile(args(1))
    val testData = spark.createDataFrame(data_process(test_rawdata), schema).collect()
    val predictions = testData.par.map { case Row(label: Long, features: Vector) =>
      val ensemblePredictions = models.map(_.predict(features))

      val predictedLabel = ensemblePredictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
      (label, predictedLabel)
    }


    def calculateAccuracy(predictions: Array[Long], labels: Array[Long]): Double = {
        val totalCount = predictions.size
        val correctCount = predictions
            .zip(labels)
            .filter(p => p._1 == p._2)
            .size
        correctCount.toDouble*100 / totalCount
    }

    val accuracy = calculateAccuracy(predictions.toArray.map(_._1), predictions.toArray.map(_._2))
    println(s"\n\nAccuracy with train samples=${train_data.count()}, k=$k, numModels=$numModels on ${testData.size} test samples : $accuracy %\n\n")

    spark.stop()

  }

  def data_process(rawdata: RDD[String]): RDD[Row] = {

    val formattedData = rawdata.map { line =>
            val parts = line.split(" ")
            val label = parts(0).toLong // Extract the label
            val features = parts.drop(2).map(_.split(":")).map { case Array(index, value) =>
                value.toDouble
            }
            Row(label, Vectors.dense(features))
    }

    formattedData
  }

  def trainKNNModel(trainingData: DataFrame, k:Int): KNN_Model = {
  val rddData = trainingData.rdd.map { case Row(label: Long, features: Vector) =>
    (label, features)
  }

  KNN_Model(k, rddData.cache())
  }

  case class KNN_Model(k: Int, trainingData: RDD[(Long, Vector)]) {
    def predict(features: Vector): Long = {
        implicit val ordering: Ordering[(Long, Double)] = Ordering.by(_._2)

        // Define a function to compute the Euclidean distance
        def euclideanDistance(v1: Vector, v2: Vector): Double = {
            math.sqrt(v1.toArray.zip(v2.toArray).map { case (x, y) => math.pow(x - y, 2) }.sum)
        }

        val neighbors = trainingData.mapValues(train_features => euclideanDistance(train_features, features))
        .takeOrdered(k)

        // Get the labels of the k-nearest neighbors
        val pred = neighbors.map(_._1)

        // Use majority voting to determine the predicted label
        val predictedLabel = pred.groupBy(identity).mapValues(_.size).maxBy(_._2)._1

        predictedLabel
    }
}





}