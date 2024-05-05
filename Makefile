# Setup if running on windows
ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
endif

# Customize these paths for your environment.
# -----------------------------------------------------------
spark.root=/usr/local/spark-3.3.2-bin-without-hadoop
hadoop.root=/usr/local/hadoop-3.3.5
app.name=CustomEnsemble
jar.name=spark-demo.jar
maven.jar.name=spark-demo-1.0.jar
job.name=ensemble.EnsembleBagging
job.name.knn=ensemble.KnnEnsemble
local.master=local[4]
local.input=input
local.output=output-local
local.train=train_data
local.test=test_data
local.numModels=4
local.k=5
local.fraction=0.7
# -1 to consider all test samples
local.test_samples=-1
local.smoothing=1.0
local.depth=5
spark.driver.memory=5g
spark.executor.memory=5g
# Pseudo-Cluster Execution
hdfs.user.name=joe
hdfs.input=input
hdfs.output=output
# AWS EMR Execution
aws.emr.release=emr-6.10.0
aws.bucket.name=cs6240-project-bucket-svs1
aws.input=input
aws.test_input=test_data
aws.output=output-aws
aws.log.dir=log
aws.num.nodes=4
aws.instance.type=m4.large
aws.subnet.id=subnet-000d3d4bd8c460602
# -----------------------------------------------------------

# Compiles code and builds jar (with dependencies).
jar:
	mvn clean package
	cp target/${maven.jar.name} ${jar.name}

# Removes local output directory.
clean-local-output:
	rm -rf ${local.output}*

# Removes local output directory in windows platform.
clean-local-output-w:
	if(Test-Path ${local.output}) { rm -Recurse -Force -erroraction 'SilentlyContinue' ${local.output} 	} else { Write-Host "No target to delete"}

clean-local-aws-output:
	rm -rf ${aws.output}*

# Runs standalone
local: jar clean-local-output
	spark-submit --class ${job.name} --master ${local.master} --name "${app.name}" ${jar.name} ${local.train} ${local.test} ${local.numModels} ${local.k} ${local.fraction}

# Runs standalone
local-knn: jar clean-local-output
	spark-submit --class ${job.name.knn} --master ${local.master} --name "${app.name}" ${jar.name} ${local.train} ${local.test} ${local.numModels} ${local.k} ${local.fraction}

# Runs standalone on windows
local-w: jar clean-local-output-w
	spark-submit --class ${job.name} --master ${local.master} --name "${app.name}" target/${jar.name} ${local.input} "train.txt" "test.txt" ${local.output}

# Start HDFS
start-hdfs:
	${hadoop.root}/sbin/start-dfs.sh

# Stop HDFS
stop-hdfs:
	${hadoop.root}/sbin/stop-dfs.sh

# Start YARN
start-yarn: stop-yarn
	${hadoop.root}/sbin/start-yarn.sh

# Stop YARN
stop-yarn:
	${hadoop.root}/sbin/stop-yarn.sh

# Reformats & initializes HDFS.
format-hdfs: stop-hdfs
	rm -rf /tmp/hadoop*
	${hadoop.root}/bin/hdfs namenode -format

# Initializes user & input directories of HDFS.
init-hdfs: start-hdfs
	${hadoop.root}/bin/hdfs dfs -rm -r -f /user
	${hadoop.root}/bin/hdfs dfs -mkdir /user
	${hadoop.root}/bin/hdfs dfs -mkdir /user/${hdfs.user.name}
	${hadoop.root}/bin/hdfs dfs -mkdir /user/${hdfs.user.name}/${hdfs.input}

# Load data to HDFS
upload-input-hdfs: start-hdfs
	${hadoop.root}/bin/hdfs dfs -put ${local.input}/* /user/${hdfs.user.name}/${hdfs.input}

# Removes hdfs output directory.
clean-hdfs-output:
	${hadoop.root}/bin/hdfs dfs -rm -r -f ${hdfs.output}*

# Download output from HDFS to local.
download-output-hdfs:
	mkdir ${local.output}
	${hadoop.root}/bin/hdfs dfs -get ${hdfs.output}/* ${local.output}

# Runs pseudo-clustered (ALL). ONLY RUN THIS ONCE, THEN USE: make pseudoq
# Make sure Hadoop  is set up (in /etc/hadoop files) for pseudo-clustered operation (not standalone).
# https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/SingleCluster.html#Pseudo-Distributed_Operation
pseudo: jar stop-yarn format-hdfs init-hdfs upload-input-hdfs start-yarn clean-local-output
	spark-submit --class ${job.name} --master yarn --deploy-mode cluster ${jar.name} ${local.input} ${local.output}
	make download-output-hdfs

# Runs pseudo-clustered (quickie).
pseudoq: jar clean-local-output clean-hdfs-output
	spark-submit --class ${job.name} --master yarn --deploy-mode cluster ${jar.name} ${local.input} ${local.output}
	make download-output-hdfs

# Create S3 bucket.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

# Upload data to S3 input dir.
upload-input-aws: make-bucket
	aws s3 sync ${local.train} s3://${aws.bucket.name}/${aws.input}

# Upload data to S3 test dir.
upload-test-data-aws: make-bucket
	aws s3 sync ${local.test} s3://${aws.bucket.name}/${aws.test_input}

# Delete S3 output dir.
delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.name} s3://${aws.bucket.name}

# Main EMR launch.
aws: jar upload-app-aws delete-output-aws
	aws emr create-cluster \
		--name "Knn Ensemble - k=${local.k}, m=${local.numModels}, testN=1000, train=830M" \
		--release-label ${aws.emr.release} \
		--ec2-attributes SubnetId=${aws.subnet.id} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster",--master,yarn,--conf,spark.yarn.submit.waitAppCompletion=true,--conf,spark.executor.memory=5g,--conf,spark.driver.memory=5g,"--class","${job.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.test_input}","${local.numModels}","${local.k}","${local.fraction}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification": "hadoop-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}, {"Classification": "spark-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate

aws-dn: jar upload-app-aws delete-output-aws
	aws emr create-cluster \
		--name "Knn Ensemble - k=${local.k}, m=${local.numModels}, testN=1000, train=830M" \
		--release-label ${aws.emr.release} \
		--ec2-attributes SubnetId=${aws.subnet.id} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster",--master,yarn,--conf,spark.yarn.submit.waitAppCompletion=true,--conf,spark.executor.memory=${spark.driver.memory},--conf,spark.driver.memory=${spark.executor.memory},"--class","${job.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.test_input}","${local.numModels}","${local.k}","${local.fraction}","${local.test_samples}","${local.smoothing}","${local.depth}","${spark.driver.memory}","${spark.executor.memory}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification": "hadoop-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}, {"Classification": "spark-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate

# Main EMR launch.
aws-knn: jar upload-app-aws delete-output-aws
	aws emr create-cluster \
		--name "Knn Ensemble - k=${local.k}, m=${local.numModels}, testN=1000, train=830M" \
		--release-label ${aws.emr.release} \
		--ec2-attributes SubnetId=${aws.subnet.id} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster",--master,yarn,--conf,spark.yarn.submit.waitAppCompletion=true,--conf,spark.executor.memory=5g,--conf,spark.driver.memory=5g,"--class","${job.name.knn}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.test_input}","${local.numModels}","${local.k}","${local.fraction}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification": "hadoop-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}, {"Classification": "spark-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate


#windows emr launch
Arguments = [\"spark-submit\",\"--deploy-mode\",\"cluster\",\"--master\",\"yarn\",\"--conf\",\"spark.yarn.submit.waitAppCompletion=true\",\"--conf\",\"spark.executor.memory=5g\",\"--conf\",\"spark.driver.memory=5g\",\"--class\",\"${job.name}\",\"s3://${aws.bucket.name}/${jar.name}\",\"s3://${aws.bucket.name}/${aws.input}\",\"s3://${aws.bucket.name}/${aws.test_input}\",\"${local.numModels}\",\"${local.k}\",\"${local.fraction}\", \"1000\"]
Steps = '[{\"Type\":\"CUSTOM_JAR\",\"Name\":\"${app.name}\",\"Jar\":\"command-runner.jar\",\"ActionOnFailure\":\"TERMINATE_CLUSTER\",\"Args\":${Arguments}}]'
InstanceGroups = '[{\"InstanceCount\":${aws.num.nodes},\"InstanceGroupType\":\"CORE\",\"InstanceType\":\"${aws.instance.type}\"},{\"InstanceCount\":1,\"InstanceGroupType\":\"MASTER\",\"InstanceType\":\"${aws.instance.type}\"}]'
Configurations = '[{\"Classification\": \"hadoop-env\", \"Configurations\": [{\"Classification\": \"export\",\"Configurations\": [],\"Properties\": {\"JAVA_HOME\": \"/usr/lib/jvm/java-11-amazon-corretto.x86_64\"}}],\"Properties\": {}}, {\"Classification\": \"spark-env\", \"Configurations\": [{\"Classification\": \"export\",\"Configurations\": [],\"Properties\": {\"JAVA_HOME\": \"/usr/lib/jvm/java-11-amazon-corretto.x86_64\"}}],\"Properties\": {}}]'
aws-w: upload-app-aws delete-output-aws
	aws emr create-cluster --name "Ensemble Spark Cluster" --release-label ${aws.emr.release} --instance-groups  ${InstanceGroups} --applications Name=Hadoop Name=Spark --steps ${Steps} --log-uri s3://${aws.bucket.name}/${aws.log.dir} --configurations ${Configurations} --use-default-roles --enable-debugging --auto-terminate


# Download output from S3.
download-output-aws: clean-local-aws-output
	mkdir ${aws.output}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${aws.output}

# Change to standalone mode.
switch-standalone:
	cp config/standalone/*.xml ${hadoop.root}/etc/hadoop

# Change to pseudo-cluster mode.
switch-pseudo:
	cp config/pseudo/*.xml ${hadoop.root}/etc/hadoop

# Package for release.
distro:
	rm -f Spark-Demo.tar.gz
	rm -f Spark-Demo.zip
	rm -rf build
	mkdir -p build/deliv/Spark-Demo
	cp -r src build/deliv/Spark-Demo
	cp -r config build/deliv/Spark-Demo
	cp -r input build/deliv/Spark-Demo
	cp pom.xml build/deliv/Spark-Demo
	cp Makefile build/deliv/Spark-Demo
	cp README.txt build/deliv/Spark-Demo
	tar -czf Spark-Demo.tar.gz -C build/deliv Spark-Demo
	cd build/deliv && zip -rq ../../Spark-Demo.zip Spark-Demo
	