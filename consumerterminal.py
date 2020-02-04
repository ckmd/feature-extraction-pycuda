from pykafka import KafkaClient

topicname = 'bustopic'
client = KafkaClient(hosts='localhost:9092')
while True:
    for i in client.topics[topicname].get_simple_consumer():
        print(format(i.value.decode()))
