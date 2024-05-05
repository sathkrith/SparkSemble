package ensemble

import ensemble.KNNEnsembleBagging.{data_process, trainKNNModel}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types.{LongType, StructField, StructType}

import scala.collection.parallel.mutable.ParArray
import scala.util.Random

/**
 * Represents a model for classification or regression.
 * Implementations should provide methods for training and prediction.
 */
trait Model2{

  /**
   * Trains the model using the provided dataset.
   *
   * @param data  The training dataset as an array of Row objects.
   */
  def predict(dataPoints: Array[(Long, linalg.Vector)]): Array[(Long,Long)]

  /**
   * Predicts the labels or values for the given data points.
   *
   * @param dataPoints  The array of data points to predict, each represented as a tuple of (id, features).
   * @return            The predicted labels or values as an array of tuples (id, label or value).
   */
  def train(data: Array[Row])
}

/**
 * Provides functionality for ensemble learning using bagging.
 * Includes methods for sampling data and training models with samples.
 */
object  EnsembleBagging {

  /**
   * Samples the data array with a given fraction and replacement option.
   *
   * @param data         The original data array to sample.
   * @param fraction     The fraction of data to sample.
   * @param withReplacement  A flag indicating whether to sample with replacement.
   * @return             The sampled data array.
   */
  private def sampleData(data: Array[Row], fraction: Double, withReplacement : Boolean  = false): Array[Row] = {
    if(withReplacement) {
      val rand = new Random(seed = Random.nextInt())
      data.map(row
      =>
        if(rand.nextFloat() >fraction)
          data(rand.nextInt(data.length))
        else
          row)
    } else {
      val rand = new Random(seed = Random.nextInt())
      data.filter(_ => rand.nextFloat() < fraction)
    }
  }

  /**
   * Trains a model using a sampled subset of the data and predicts the labels for test data.
   *
   * @param data         The original data array.
   * @param testdata     The test data as an array of (id, features) tuples.
   * @param fraction     The fraction of data to sample.
   * @param model        The model to train and use for prediction.
   * @return             The predicted labels as an array of (id, label) tuples.
   */
  private def trainModelWithSample(data: Array[Row], testdata: Array[(Long, Vector)], fraction:Double , model: Model2): Array[(Long,Long)] = {
    val sampledData = sampleData(data,fraction, withReplacement = false)
    model.train(sampledData)
    model.predict(testdata).toArray
  }

  /**
   * Main method for running the ensemble bagging algorithm.
   *
   * @param args  Command line arguments specifying the input directories, number of models, etc.
   */
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 10) {
      logger.error("Usage:\nensemble.Ensemble <train data dir> <test data dir> numModels K fraction num_test_samples num_copies")
      logger.error("Example:\nensemble.Ensemble data/train data/test 2 3 0.5")
      logger.error(args.mkString(" "))
      System.exit(1)
    }
    val numModels = args(2).toInt
    val k = args(3).toInt
    val fraction = args(4).toDouble
    val train_files = args(0)
    val test_files = args(1)
    val num_test_samples = args(5).toInt
    val smoothing = args(6).toDouble
    val depth = args(7).toInt
    val driverMemory = args(8)
    val executorMemory = args(9)
    val metaData = MetaDataExtractor.getMetaData()
    logger.setLevel(Level.ERROR)
    var index = 0
    val spark = SparkSession.builder()
      .appName("DecisionTreeEnsemble")
      .config("spark.driver.memory", driverMemory)
      .config("spark.executor.memory", executorMemory)
     // .master("local[*]") // Use local for testing, specify your cluster manager in a real environment
      .getOrCreate()
    val sc = spark.sparkContext
    val schema = new StructType()
      .add(StructField("label", LongType, false))
      .add(StructField("features", VectorType, false))
    val train_rawdata = sc.textFile(train_files)
    val test_rawdata = sc.textFile(test_files)
    import spark.implicits._
    val trainData =  sc.broadcast(spark.createDataFrame(data_process(train_rawdata), schema).collect())
    val testData = spark.createDataFrame(data_process(test_rawdata), schema).withColumn("row_index", monotonically_increasing_id())
    val testDataRdd = testData.map(row => (row.getAs[Long]("row_index"), row.getAs[Long]("label"), row.getAs[Vector]("features")))
    val testDataFinal = if(num_test_samples > 0) testDataRdd.limit(num_test_samples).collect() else testDataRdd.collect()

    val test_featuresWithIndex = testDataFinal.map(
      row => (row._1, row._3)
    )

    // Create and distribute model tasks
    val models = Seq.fill(numModels / 2)(new DecisionTree2(depth, metaData)) ++ Seq.fill(numModels / 2)(new MultinodalNaiveBayesModel(smoothing, metaData))
    val modelTasks = spark.sparkContext.parallelize(models, numSlices = numModels)

    // Train models with unique samples
    val predictions = modelTasks.map(model => {
      trainModelWithSample(trainData.value,test_featuresWithIndex,fraction,  model)
    })

    val test_labelsWithIndex= testDataFinal.map(row => (row._1, row._2))
    // Flatten the ParArray to merge all RDDs into one RDD
    val flatPredictions= predictions.flatMap(_.toSeq)

    // Group the RDD by the test sample index
    val groupedPredictions = flatPredictions.groupBy(_._1)

    // For each group, count the occurrences of each class label
    val countedPredictions: RDD[(Long, Map[Long, Int])] = groupedPredictions.mapValues(_.groupBy(_._2).mapValues(_.size))

    // Select the class label with the highest count for each test sample index
    val finalPredictions = countedPredictions.mapValues(_.maxBy(_._2)._1).collect()
    val totalValues = finalPredictions.length
    val pred_true_classes = finalPredictions.map(pred => (pred._2, test_labelsWithIndex.find(_._1 == pred._1).get._2))

    val countEqual = pred_true_classes.filter { case ((pred, label)) => pred == label }.length

    //pred_true_classes.saveAsTextFile("output")

    println(s"\n\nAccuracy with train samples=${trainData.value.length}, depth = ${depth}, smoothness = ${smoothing}, numModels=$numModels, bootstrapFraction=$fraction on ${totalValues} test samples : ${countEqual * 100 / totalValues} %\n\n")

    spark.stop()
  }
}