package ensemble

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.rdd.RDD


object ParallelKNNEnsemble {
  def main(args: Array[String]): Unit = {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 5) {
      logger.error("Usage:\nensemble.ParallelKNNEnsemble <train data dir> <test data dir> numModels K fraction")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("Parallel KNN Ensemble")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().config(conf).getOrCreate()

    val numModels = args(2).toInt
    val k = args(3).toInt
    val fraction = args(4).toDouble

    val rawdata = spark.sparkContext.textFile(args(0))


    val formattedData = rawdata.map { line =>
            val parts = line.split(" ")
            val label = parts(0).toLong // Extract the label
            val features = parts.drop(2).map(_.split(":")).map { case Array(index, value) =>
                value.toDouble
            }
            (label, features)
    }.cache()
    
    // val schema = new StructType()
    //             .add(StructField("features", ArrayType(DoubleType), false))
    //             .add(StructField("label", DoubleType, false))
    
    // val data = spark.createDataFrame(formattedData, schema)
    // data.show(5)

    // Function to calculate Euclidean distance
    def euclideanDistance(v1: Array[Double], v2: Array[Double]): Double = {
        math.sqrt(v1.zip(v2).map { case (x1, x2) => math.pow(x1 - x2, 2) }.sum)
    }

    // KNN prediction for a single instance
    def knnPredict(trainingData: RDD[(Long, Array[Double])], queryPoint: Array[Double], k: Int): Long = {
        implicit val ordering: Ordering[(Long, Double)] = Ordering.by(_._2)

        // (label, distance)
        val neighbors = trainingData.mapValues(features => euclideanDistance(features, queryPoint)).takeOrdered(k)
        // print("\n\n printing neighbors \n\n")
        // neighbors.foreach(println)

        // For classification, return the most common label among the neighbors
        // neighbors.map(_._2).groupBy(identity).mapValues(_.size).maxBy(_._2)._1
        neighbors.map(_._1).groupBy(identity).mapValues(_.size).maxBy(_._1)._1

    }

    val model = (query: Array[Double]) => knnPredict(formattedData, query, k)

    val test = formattedData.takeSample(false, 100)

    val preds = test.par.map{case (label, query) => (label, model(query))}
    // val preds = test.map{case (label, query) => (label, model(query))}

    // preds.foreach(println)

    // val preds = test.foreach(model)

    // val pred = knnPredict(formattedData, test._2, k)
    // print(s"\n\nPrediction - $pred. Actual - ${test._1}\n\n")

    // val rawdata_test = spark.sparkContext.textFile(args(1))

    // val formattedData_test = rawdata_test.map { line =>
    //         val parts = line.split(" ")
    //         val label = parts(0).toLong // Extract the label
    //         val features = parts.drop(2).map(_.split(":")).map { case Array(index, value) =>
    //             value.toDouble
    //         }
    //         (label, features)
    // }



    // val preds = formattedData.take(100).map{case (label, features) => 
    //   val prediction = knnPredict(formattedData, features, k)
    //   (prediction, label)
      
    // }

    def calculateAccuracy(predictions: Array[Long], labels: Array[Long]): Double = {
        val totalCount = predictions.size
        val correctCount = predictions
            .zip(labels)
            .filter(p => p._1 == p._2)
            .size
        correctCount.toDouble*100 / totalCount
    }

    val accuracy = calculateAccuracy(preds.toArray.map(_._1), preds.toArray.map(_._2))
    // val accuracy = calculateAccuracy(preds.map(_._1), preds.map(_._2))
    println(s"\n\nAccuracy with k=$k, numModels=1 : $accuracy %\n\n")
    println(formattedData.toDebugString)
    // predictions.foreach(println)

    sc.stop()

    
}
}