import json
from datetime import datetime
from pykafka import KafkaClient

client = KafkaClient(hosts="localhost:9092")

topic = client.topics['bustopic']

producer = topic.get_sync_producer()

input_file = open('rute1.json')
json_array = json.load(input_file)
coordinates = json_array['features'][0]['geometry']['coordinates']
print(coordinates)

def generate_uuid():
    return uuid.uuid4()

data = {}
data['busline'] = '001'

def generate_checkpoint(coordinates):
    i = 0
    while i < len(coordinates):
        data['key'] = data['busline'] + '_' + str(generate_uuid)
        data['timestamp'] = str(datetime.utcnow())
        data['latitude'] = coordinates[i][0]
        data['longitude'] = coordinates[i][1]
        message = json.dumps(data)
        producer.produce(message.encode('ascii'))
        print(message)
        if(i == len(coordinates)-1):
            i = 0
        else:
            i += 1

generate_checkpoint(coordinates)