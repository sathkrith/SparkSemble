package ensemble

import org.apache.spark.ml.linalg.Vector

final case class DataPoint(features: Vector, label: Double)
