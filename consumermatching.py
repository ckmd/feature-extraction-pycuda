from pykafka import KafkaClient
import json, pandas, pickle, numpy, time
from dbconnect import connection

print("Initializing ... ")
topicname = 'match5'
client = KafkaClient(hosts='localhost:9092')
consumer = client.topics[topicname].get_simple_consumer()
c, conn = connection()

def sigmoid(x):
    return 1/(1+numpy.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# load trained model
weights = pickle.load(open("model/5labelsyn0.pickle", "rb"))
weights2 = pickle.load(open("model/5labelsyn1.pickle", "rb"))
bias = pickle.load(open("model/5labelbias.pickle", "rb"))
bias2 = pickle.load(open("model/5labelbias2.pickle", "rb"))

lr = 0.05
iterration = 1
datake = 1
accuracy = 0
identity = 36
temp = 1000
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
            if(z[0][indeks] >= 0.75):
                if(temp != indeks):
                    temp = indeks
                else:
                    if(temp == indeks):
                        print(indeks, z[0][indeks], valjson['timestamp'])
                    temp = 1000
                # identity += 1
                # sql = "INSERT INTO log (id,user_id,jetson_id, recorded_time) VALUES (%s,%s,%s,%s)"
                # val = (identity,indeks,int(valjson['jetsonid']),valjson['timestamp'])
                # c.execute(sql,val)
                # conn.commit()
