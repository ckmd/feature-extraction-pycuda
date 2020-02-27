from pykafka import KafkaClient
import json, pandas, pickle, numpy, time
from dbconnect import connection

# print("Initializing ... ")
topicname = 'matchtopic1'
client = KafkaClient(hosts='localhost:9092')
consumer = client.topics[topicname].get_simple_consumer()
c, conn = connection()

def sigmoid(x):
    return 1/(1+numpy.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# load trained model
weights = pickle.load(open("model/2syn0.pickle", "rb"))
weights2 = pickle.load(open("model/2syn1.pickle", "rb"))
bias = pickle.load(open("model/2bias.pickle", "rb"))
bias2 = pickle.load(open("model/2bias2.pickle", "rb"))

# Defining Neural Network Synapse for the first time
# numpy.random.seed(0)
# weights = 2 * numpy.random.rand(2176,900) - 1
# weights2 = 2 * numpy.random.rand(900,90) - 1
# bias = 2 * numpy.random.rand(1,900) - 1
# bias2 = 2 * numpy.random.rand(1,90) - 1
lr = 0.05
iterration = 1
datake = 1
accuracy = 0
identity = 36
print("waiting for message")
while True:
    for message in consumer:
        if message is not None:
            start = time.time()
            val = message.offset, message.value
            valjson = json.loads(val[1])
            # convert the string:string dictionary into int:float
            # sorting the value by index, so the data will not randomly structured
            features = {int(k):float(v) for k,v in valjson['features'].items()}
            # restructure dict into array
            inputlayer = numpy.array([features.values()])
            # forward propagation
            l1 = sigmoid(numpy.dot(inputlayer, weights) + bias)
            z = sigmoid(numpy.dot(l1, weights2) + bias2)
            indeks = numpy.argmax(z)
            # if(z[0][indeks] >= 0.5):
            identity += 1
            print(indeks, z[0][indeks])
                # sql = "INSERT INTO log (id,user_id,jetson_id, recorded_time) VALUES (%s,%s,%s,%s)"
                # val = (identity,indeks,int(valjson['jetsonid']),valjson['timestamp'])
                # c.execute(sql,val)
                # conn.commit()
