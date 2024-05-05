package ensemble

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.spark.ml.linalg
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

import java.net.{InetAddress, NetworkInterface}
import scala.collection.parallel.mutable.ParArray
import scala.util.Random

trait model{
  def predict(dataPoints: Array[(Long,Vector)]): ParArray[(Long,Long)]
}

object KNNEnsembleBagging {
  def main(args: Array[String]): Unit = {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 5) {
      logger.error("Usage:\nensemble.KNNEnsembleBagging <train data dir> <test data dir> numModels K fraction")
      logger.error("Example:\nensemble.KNNEnsembleBagging data/train data/test 2 3 0.5")
      logger.error(args.mkString(" "))
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

    val localHost = InetAddress.getLocalHost
    val ni = NetworkInterface.getByInetAddress(localHost)
    val hardwareAddress = ni.getHardwareAddress.mkString("Array(", ", ", ")")
    logger.info("Running ensemble in:" + hardwareAddress)

    val train_rawdata = spark.sparkContext.textFile(args(0))
    val metaData = MetaDataExtractor.getMetaData()
    
    val train_data = spark.createDataFrame(data_process(train_rawdata), schema)
    sc.broadcast(train_data)

    val map: Map[Int, String] = Map((1, "DecisionTree"), (2, "KNN"))

    val models = (1 until numModels+1).par.map { index =>
      val bootstrapData = train_data.sample(withReplacement = true, fraction = fraction, seed = Random.nextInt())
      if(map(index) == "DecisionTree"||map(index) == "KNN"){
        val dTreeModel = new DecisionTree(bootstrapData, maxDepth =  5, metaData = metaData)
        dTreeModel.train()
        dTreeModel
      }
      else{
        val model = trainKNNModel(bootstrapData, k = k)
        model
      }
    }.toArray

    val test_rawdata = spark.sparkContext.textFile(args(1))
    val testData = spark.createDataFrame(data_process(test_rawdata), schema).withColumn("row_index", monotonically_increasing_id()).collect()
    

    val test_featuresWithIndex: Array[(Long, Vector)] = testData.map(
      row => (
        row.getAs[Long]("row_index"),
        row.getAs[linalg.Vector]("features").toDense
        )
    )
    val test_labelsWithIndex: Array[(Long, Long)] = testData.map(row => (row.getAs[Long]("row_index"), row.getAs[Long]("label")))

    val predictions = models.par.map{model => 
        spark.sparkContext.parallelize(model.predict(test_featuresWithIndex).seq)
    }
    // Flatten the ParArray to merge all RDDs into one RDD
    val flatPredictions: RDD[(Long, Long)] = spark.sparkContext.union(predictions.seq)

    // Group the RDD by the test sample index
    val groupedPredictions: RDD[(Long, Iterable[(Long, Long)])] = flatPredictions.groupBy(_._1)

    // For each group, count the occurrences of each class label
    val countedPredictions: RDD[(Long, Map[Long, Int])] = groupedPredictions.mapValues(_.groupBy(_._2).mapValues(_.size))

    // Select the class label with the highest count for each test sample index
    val finalPredictions: RDD[(Long, Long)] = countedPredictions.mapValues(_.maxBy(_._2)._1)

    val pred_true_classes = finalPredictions.join(spark.sparkContext.parallelize(test_labelsWithIndex.seq))
    
    val countEqual = pred_true_classes.filter{ case (_, (pred, label)) => pred == label}.count()
    
    // println(pred_true_classes.toDebugString)
    
    println(s"\n\nAccuracy with train samples=${train_data.count()}, k=$k, numModels=$numModels, bootstrapFraction=$fraction on ${testData.size} test samples : ${countEqual*100/testData.size} %\n\n")

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

  def trainKNNModel(trainingData: DataFrame, k:Int): KNNModel = {
  val rddData = trainingData.rdd.map { case Row(label: Long, features: Vector) =>
    (label, features)
  }
  KNNModel(k, rddData)
  }


}
