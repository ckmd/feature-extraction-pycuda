import json
from datetime import datetime
from pykafka import KafkaClient

client = KafkaClient(hosts="localhost:9092")

topic = client.topics['bustopic2']

producer = topic.get_sync_producer()

# coordinates = json_array['features'][0]['geometry']['coordinates']


data = {}
data['jetsonid'] = '001'
data['features'] = {}

def produce():
    while True:
        input_file = open('data33.json')
        features = json.load(input_file)
        # data['key'] = data['jetsonid'] + '_' + str(generate_uuid)
        data['timestamp'] = str(datetime.utcnow())
        for i in range(len(features)):
            data['features'][str(i+1)] = features[str(i+1)]
        message = json.dumps(data)
        producer.produce(message.encode('ascii'))
        print(data['timestamp'])

def generate_uuid():
    return uuid.uuid4()
produce()