# feature-extraction-pycuda
supporting project for face recognition

1. how to run feature-extraction
  > run extraction.py
2. how to run zookeeper server
  > bin/zookeeper-server-start.sh config/zookeeper.properties
3. run apache kafka
  > bin/kafka-server-start.sh config/server.properties
4. create a topic (optional if the topic doesn't exist)
  > bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic [topic name]
5. run producer / consumer
  > run producer.py
  > run consumer.py
6. run neural network mode for training or matching
  > run neuralnetwork[mode].py
