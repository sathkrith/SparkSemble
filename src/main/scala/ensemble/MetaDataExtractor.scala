package ensemble


/**
 * Contains metadata about the dataset
 */
object MetaDataExtractor{
  case class MetaData(isCategorical: Boolean, isOrdinal: Boolean, numDistinctValues: Int, max: Double, min: Double, bucketSize:Int=0, values:Seq[Double]=Seq.empty[Double])
  def getMetaData(): Seq[MetaData] = {
    val meta_data = "1,1,18,0.0,75.0\n1,1,8,0.0,7.0\n1,1,14,0.0,15.0\n1,1,13,0.0,15.0\n1,1,18,0.0,75.0\n1,1,56,0.0,1.0\n1,1,33,0.0,1.0\n1,1,46,0.0,1.0\n1,1,39,0.0,1.0\n1,1,56,0.0,1.0\n0,0,5626,0.0,12921.0\n1,1,200,0.0,3233.0\n1,1,463,0.0,6669.0\n1,1,97,2.0,265.0\n0,0,5713,2.0,13347.0\n0,0,5619,-1.939777,81.729559\n0,0,5225,3.725981,513.52506\n0,0,5324,1.993372,497.332786\n0,0,5320,2.001239,362.456874\n0,0,5620,-1.944837,81.69289\n1,1,666,0.0,338325.0\n1,1,80,0.0,178.0\n1,1,121,0.0,862.0\n1,1,17,0.0,52.0\n1,1,690,0.0,338325.0\n1,1,263,0.0,4511.0\n1,1,42,0.0,86.0\n1,1,38,0.0,315.0\n1,1,9,0.0,15.0\n1,1,271,0.0,4511.0\n1,1,507,0.0,4511.0\n1,1,49,0.0,94.0\n1,1,102,0.0,856.0\n1,1,12,0.0,37.0\n1,1,529,0.0,4511.0\n0,0,2076,0.0,4511.0\n1,1,190,0.0,86.0\n1,1,255,0.0,315.0\n1,1,70,0.0,26.0\n0,0,2136,0.0,4511.0\n0,0,15683,0.0,2763906.25\n1,1,375,0.0,2209.0\n1,1,502,0.0,161696.888889\n1,1,106,0.0,121.0\n0,0,16387,0.0,2783892.25\n0,0,75606,0.0,60.338346\n1,1,930,0.0,3.0\n0,0,1212,0.0,3.0\n1,1,262,0.0,1.636364\n0,0,80077,0.0,60.072655\n0,0,34807,0.0,1.0\n1,1,526,0.0,1.0\n1,1,629,0.0,1.0\n1,1,122,0.0,0.6\n0,0,37177,0.0,0.800969\n0,0,57535,0.0,1.0\n1,1,713,0.0,1.0\n1,1,983,0.0,1.0\n1,1,170,0.0,0.6\n0,0,60160,0.0,0.800969\n0,0,55826,0.0,1.0\n0,0,1172,0.0,1.0\n0,0,1605,0.0,1.0\n1,1,465,0.0,0.6\n0,0,59101,0.0,0.800969\n0,0,7289,0.0,0.25\n0,0,1814,0.0,0.25\n0,0,2338,0.0,0.25\n1,1,731,0.0,0.09\n0,0,7448,0.0,0.0625\n0,0,378218,-1388.740897,368557.378509\n0,0,22853,0.0,1413.027182\n0,0,42835,0.0,6927.854528\n0,0,19708,0.0,301.683641\n0,0,400182,-1390.351776,358986.728374\n0,0,74147,-718.788035,8667.005515\n0,0,5505,0.0,695.932703\n0,0,6711,0.0,2058.949308\n0,0,3470,0.0,83.650073\n0,0,79698,-720.139889,8674.545513\n0,0,148565,0.0,25963.491963\n0,0,11616,0.0,771.111504\n0,0,14185,0.0,6857.293941\n0,0,7382,0.0,218.033568\n0,0,157300,0.0,26017.674936\n0,0,381354,-281.267731,13048.199883\n0,0,25911,0.0,706.513591\n0,0,47069,0.0,2580.061807\n0,0,23219,0.0,150.84182\n0,0,403379,-287.835816,13075.29137\n0,0,354797,0.0,166804769.503892\n0,0,24181,0.0,148653.237799\n0,0,45595,0.0,10343023.179426\n0,0,22178,0.0,4514.730882\n0,0,374636,0.0,167505292.377452\n1,1,2,0.0,1.0\n1,1,2,0.0,1.0\n1,1,2,0.0,1.0\n1,1,2,0.0,1.0\n1,1,2,0.0,1.0\n0,0,197646,0.0,1.0\n0,0,16851,0.0,1.0\n0,0,34164,0.0,1.0\n0,0,19598,0.0,1.0\n0,0,201721,0.0,1.0\n0,0,580729,-13.502947,285.027752\n0,0,52085,0.0,65.075948\n0,0,159162,0.0,84.585654\n0,0,75587,0.0,72.19844\n0,0,623842,-13.539601,277.641596\n0,0,577749,-160.149259,0.0\n0,0,61098,-141.712421,0.0\n0,0,202254,-146.691595,0.0\n0,0,101066,-162.903042,0.0\n0,0,618409,-160.344508,0.0\n0,0,578061,-157.053469,0.0\n0,0,61722,-138.203961,0.0\n0,0,186909,-143.962503,0.0\n0,0,97946,-161.336069,0.0\n0,0,619763,-157.276875,0.0\n0,0,554227,-163.220923,0.0\n0,0,48799,-169.913234,0.705644\n0,0,151480,-174.793032,0.705595\n0,0,81165,-190.997484,0.0\n0,0,588318,-163.438492,0.0\n1,1,26,1.0,97.0\n1,1,389,4.0,1549.0\n0,0,22158,-2083777989.0,314131554.0\n1,1,134,0.0,178.0\n0,0,64527,100.0,65534.0\n0,0,59944,1.0,65535.0\n1,1,254,1.0,254.0\n1,1,255,0.0,254.0\n0,0,3847,0.0,13150188.0\n0,0,3356,0.0,2789632.0\n0,0,62726,0.0,980000001.0"
    val lines = meta_data.split("\n")
    lines.map( value =>{
      val values = value.split(",")
      val isCategorical = values(0).toInt == 1
      val isOrdinal = values(1).toInt == 1
      val numDistinctValues = values(2).toInt
      val max = values(4).toDouble
      val min = values(3).toDouble
      if(!isCategorical)
        MetaData(isCategorical, isOrdinal, numDistinctValues, max, min, numDistinctValues/1000+1)
      else
        MetaData(isCategorical, isOrdinal, numDistinctValues, max, min)
    })
  }
}