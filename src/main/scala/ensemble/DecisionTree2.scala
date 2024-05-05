package ensemble

import ensemble.MetaDataExtractor.MetaData
import org.apache.log4j.LogManager
import org.apache.spark.ml.linalg
import org.apache.spark.sql.{Dataset, Row}

import java.net.{InetAddress, NetworkInterface}
import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.parallel.mutable.ParArray
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._
import scala.concurrent.duration._
import scala.io.Source
import scala.util.Random

case class DecisionTree2(maxDepth: Int, var metaData: Seq[MetaData]) extends Model2{
  private var treeNode: TreeNode = null
  private var data: List[DataPoint] = null

  private case class DataPoint(features: linalg.Vector, label: Long)
  private case class TestDataPoint(index: Long, features: linalg.Vector)

  private case class TestDataPoint2(features: linalg.Vector)
  case class TreeNode(predicate: Option[(Double, Int) => Boolean] = None, left: TreeNode, right: TreeNode, prediction: Option[Long] = None, end: Boolean = false)

  /**
   * Calculate the gini impurity of the data points
   * Gini Impurity measures how well the labels are split when a data point is split on a feature
   * Lower the gini impurity, better the split
   * @param dataPoints Data points
   * @return Gini impurity
   */
  private def giniImpurity(dataPoints: Seq[DataPoint]): Double = {
    val total = dataPoints.length.toDouble
    val counts = dataPoints.groupBy(_.label).mapValues(_.size.toDouble)
    1 - counts.map { case (_, count) => math.pow(count / total, 2) }.sum
  }

  /**
   * Calculate the gini impurity on the data points after splitting on a feature
   * @param dataPoints Data points
   * @return Gini impurity
   */
  private def fastImpurityCalc(dataPoints: Seq[DataPoint], featureIndex: Int, value: Double): Double = {
    // Impurity of values less than or equal to the split value
    val impurityLeft = mutable.ArrayBuffer[Long](0,0, 0, 0, 0)
    // Impurity of values greater than the split value
    val impurityRight =  mutable.ArrayBuffer[Long](0,0, 0, 0, 0)
    dataPoints.foreach { dp =>
      if (dp.features(featureIndex) <= value) {
        impurityLeft(dp.label.toInt) = impurityLeft(dp.label.toInt) + 1
      } else {
        impurityRight(dp.label.toInt) = impurityRight(dp.label.toInt) + 1
      }
    }
    val totalLeft = impurityLeft.sum.toDouble
    val giniLeft = 1.0 - impurityLeft.map(count => math.pow(count.toDouble / totalLeft, 2)).sum
    val totalRight = impurityRight.sum.toDouble
    val giniRight = 1.0 - impurityRight.map(count => math.pow(count.toDouble / totalRight, 2)).sum
    (giniLeft * totalLeft + totalRight * giniRight) / (totalLeft + totalRight)
  }

  private def split(dataPoints: Seq[DataPoint], featureIndex: Int, value: Double): (Seq[DataPoint], Seq[DataPoint]) = {
    dataPoints.partition(dp => dp.features(featureIndex) <= value)
  }

  /**
   * Find the best split for the data points, uses parallelism to speed up the process.
   * @param dataPoints Data points
   * @param features Features to consider for splitting
   * @return Best feature, best value to split on and the gini impurity
   */
  private def findBestSplit(dataPoints: Seq[DataPoint], features: Seq[Int] ): (Int, Double, Double) = {
    var bestGini = Double.MaxValue
    var bestFeature = 0
    var bestValue = 0.0
    features.par.foreach { feature =>
      var range: Seq[Double] = metaData(feature).values
      // If the number of values in the range is greater than 20, we take 20 values from the range
      if(range.length > 20) {
        val offset = 1 + Random.nextInt(20)
        range = for (step <- Range(offset, range.length, range.length/20)) yield (range(step))
      }
      for(rangeValue <- range){
        val gini = fastImpurityCalc(dataPoints, feature, rangeValue)
        synchronized(
        if (gini < bestGini) {
          bestGini = gini
          bestFeature = feature
          bestValue = rangeValue
        })
      }
    }
    (bestFeature, bestValue, bestGini)
  }

  private def buildTree(dataPoints: Seq[DataPoint], depth: Int = 0, features: Seq[Int]): TreeNode = {
    if (depth == maxDepth || giniImpurity(dataPoints) == 0) {
      // If the depth is equal to the max depth or the gini impurity is 0, we cannot split further. Return the majority label
      TreeNode(prediction = Some(dataPoints.groupBy(_.label).maxBy(_._2.size)._1), left = null, right = null, end = true)
    } else {
      // Split the tree further. Build trees in Parallel.
      val (feature, value, _) = findBestSplit(dataPoints,features)
      val (leftSplit, rightSplit) = split(dataPoints, feature, value)
      val left = Future(buildTree(leftSplit, depth + 1, features))
      val right = Future(buildTree(rightSplit, depth + 1, features))
      TreeNode(
        predicate = Some((fv: Double, index: Int) => index == feature && fv <= value),
        left = Await.result(left, Duration.Inf),
        right = Await.result(right, Duration.Inf),
      )
    }
  }

  /**
   * Converts real valued features to bucketed features
   * @param dataPointsIterator Iterator of DataPoints
   * @param metaDataSeq Metadata of the dataset
   * @return Iterator of DataPoints with real valued features bucketed
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

  private def TransformData(dataPoints: Array[(Long, linalg.Vector)]): Array[TestDataPoint] = {
    val newDataPoints = dataPoints.map(value => {
      val newFeatures = Array.range(0, value._2.size).map(i => {
        val feature = value._2(i)
        val valueMetaData = metaData(i)
        if(!valueMetaData.isCategorical)
        {
          val bucketSize = valueMetaData.numDistinctValues/1000
          val divider = (valueMetaData.max - valueMetaData.min)/bucketSize
          val bucket = math.floor(feature/divider).toInt
          bucket.toDouble
        }
        else {
          feature
        }

      })
      TestDataPoint(index = value._1, features = linalg.Vectors.dense(newFeatures))
    })
    newDataPoints
  }

  /**
   * Train the model
   */
  def train(dataset: Array[Row]): Unit = {
    val features = Seq.range(0, metaData.length)
    val localHost = InetAddress.getLocalHost
    val ni = NetworkInterface.getByInetAddress(localHost)
    val hardwareAddress = ni.getHardwareAddress.mkString("Array(", ", ", ")")
    //logger.info("Reached decision tree with " + dataset.isLocal + " is local. Running collect in:" + hardwareAddress)
    val tempData = dataset.map(row => (row.getLong(0), row.getAs[linalg.Vector](1)))
   // logger.info("Training the model with " + tempData.length + " data points.")
    this.data = UpdateMetadata(tempData)
    treeNode = buildTree(this.data, 0, features)
  }

  /**
   * Predict the label of the data points
   * @param treeNode
   * @param dataPoint
   * @return
   */
  @tailrec
  private final def predictDataPoint(treeNode: TreeNode, dataPoint: TestDataPoint): Long = {
    // If we have reached the end of the tree, return the prediction of the tree node
    if(treeNode.end)
      return treeNode.prediction.get
    treeNode.predicate match {
      case Some(pred) =>{
        var foundLeft = false;
        for(i <- Range(0, dataPoint.features.size))
          if(pred(dataPoint.features(i), i))
            foundLeft = true
        if(foundLeft)
          predictDataPoint(treeNode.left, dataPoint)
        else
          predictDataPoint(treeNode.right, dataPoint)
      }
      case None => treeNode.prediction.get
    }
  }

  @tailrec
  private final def predictDataPoint2(treeNode: TreeNode, dataPoint: TestDataPoint2): Long = {
    if (treeNode.end)
      return treeNode.prediction.get
    treeNode.predicate match {
      case Some(pred) => {
        var foundLeft = false;
        for (i <- Range(0, dataPoint.features.size))
          if (pred(dataPoint.features(i), i))
            foundLeft = true
        if (foundLeft)
          predictDataPoint2(treeNode.left, dataPoint)
        else
          predictDataPoint2(treeNode.right, dataPoint)
      }
      case None => treeNode.prediction.get
    }
  }

  private def TransformDataPoint(dataPoint:  linalg.Vector): TestDataPoint2 = {

      val newFeatures = Array.range(0, dataPoint.size).map(i => {
        val feature = dataPoint(i)
        val valueMetaData = metaData(i)
        if (!valueMetaData.isCategorical) {
          val bucketSize = valueMetaData.numDistinctValues / 1000
          val divider = (valueMetaData.max - valueMetaData.min) / bucketSize
          val bucket = math.floor(feature / divider).toInt
          bucket.toDouble
        }
        else {
          feature
        }

      })
      TestDataPoint2(features = linalg.Vectors.dense(newFeatures))
    }

  def predict(dataPoints: Array[(Long, linalg.Vector)]): Array[(Long,Long)] = {
    TransformData(dataPoints).par.map(point => (point.index, predictDataPoint(treeNode, point))).toArray
  }

  def predictPoint(dataPoint:  linalg.Vector): Long = {
    val transformedPoint = TransformDataPoint(dataPoint)
    predictDataPoint2(treeNode, transformedPoint)
  }

  private def processLine(line: String): (Int, Vector[Double], Long) = {
    val featureValuePairsList = line.split(" ")
    val key = featureValuePairsList(0)
    val qid = featureValuePairsList(1).split(":")(1).toLong
    val featureList = (2 until featureValuePairsList.length).map { i =>
      val pair = featureValuePairsList(i).split(":")
      pair(1).toDouble
    }.toVector
    (key.toInt, featureList, qid)
  }

  private def loadDataInParallel(path:String):List[(Int, Vector[Double], Long)] = {
    val textFile =Source.fromFile(path)
    val batchSize = 1000 // Number of lines to read at a time
    val linesIterator = textFile.getLines()
    var lineBatch = linesIterator.take(batchSize).toSeq
    var trainData:List[(Int, Vector[Double], Long)] = List.empty[(Int, Vector[Double], Long)]
    while (lineBatch.nonEmpty) {
      // Process each batch in parallel
      val futures = lineBatch.par.map(line => Future(processLine(line)))
      trainData = trainData.++(futures.map(Await.result(_, Duration.Inf)).toList)
      lineBatch = linesIterator.take(batchSize).toSeq
    }
    trainData
  }
  def main(args: Array[String]): Unit = {
    val input_dir = args(0)
    val train_files = args(1).split(",")
    val test_files = args(2).split(",")
    val output = args(3)
    val trainData:List[(Int, Vector[Double], Long)] = loadDataInParallel(input_dir + "/" + train_files(0))
    val testData :List[(Long, Int, Vector[Double], Long)] = loadDataInParallel(input_dir + "/" + test_files(0))
      .zipWithIndex.map(x => (x._2.toLong, x._1._1, x._1._2, x._1._3))
    //val metaData = MetaDataExtractor.getMetaData("meta_data.txt", input_dir)

    //val dc = DecisionTree(trainData, 5, metaData)
    //train(trainData.iterator)

    //var success = 0
    //var failure = 0
    //val predictions = predict(testData.toIterator)

   //print("Accuracy:", success.toDouble/(success.toDouble + failure.toDouble))
  }
}