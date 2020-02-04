from flask import Flask, Response
from pykafka import KafkaClient

app = Flask(__name__)

@app.route('/<topicname>')
def get_messages(topicname):
    client = KafkaClient(hosts='localhost:9092')
    def events():
        for i in client.topics[topicname].get_simple_consumer():
            yield 'data:{0}\n\n'.format(i.value.decode())
    return Response(events(), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, port=5001)