from pykafka import KafkaClient
import json, pandas, numpy

topicname = 'bustopic2'
client = KafkaClient(hosts='localhost:9092')
consumer = client.topics[topicname].get_simple_consumer()
while True:
    for message in consumer:
        if message is not None:
            val = message.offset, message.value
            valjson = json.loads(val[1])
            # convert the string:string dictionary into int:float
            valjson['features'] = {int(k):float(v) for k,v in valjson['features'].items()}
            inputlayer = numpy.array([valjson['features'].values()])
            # sipp... input layer sudah sesuai dengan neural networknya
            print(inputlayer[0][0])