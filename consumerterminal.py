from pykafka import KafkaClient
import json, pandas, pickle, numpy

# print("Initializing ... ")
topicname = 'bustopic3'
client = KafkaClient(hosts='localhost:9092')
consumer = client.topics[topicname].get_simple_consumer()

def sigmoid(x):
    return 1/(1+numpy.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Defining Neural Network Synapse
numpy.random.seed(0)
weights = 2 * numpy.random.rand(2176,1000) - 1
weights2 = 2 * numpy.random.rand(1000,90) - 1
bias = 2 * numpy.random.rand(1,1000) - 1
bias2 = 2 * numpy.random.rand(1,90) - 1
lr = 0.05
iterration = 1

print("waiting for message")
while True:
    for message in consumer:
        if message is not None:
            val = message.offset, message.value
            valjson = json.loads(val[1])
            # convert the string:string dictionary into int:float
            # sorting the value by index, so the data will not randomly structured
            features = {int(k):float(v) for k,v in valjson['features'].items()}
            label = {int(k):float(v) for k,v in valjson['label'].items()}
            # restructure dict into array
            inputlayer = numpy.array([features.values()])
            labellayer = numpy.array([label.values()])
            # forward propagation
            l1 = sigmoid(numpy.dot(inputlayer, weights) + bias)
            z = sigmoid(numpy.dot(l1, weights2) + bias2)
            # backpropagation step 1
            error = z - labellayer
            print(z)
            # backpropagation step 2
            dcost_dpred = error
            dpred_dz = sigmoid_der(z)
            z_delta = dcost_dpred * dpred_dz

            l1_error = z_delta.dot(weights2.T)
            dpred_dl1 = sigmoid_der(l1_error)
            l1_delta = l1_error * dpred_dl1

            l1 = l1.T
            weights2 -= lr * numpy.dot(l1,z_delta)
            inputlayer = inputlayer.T
            weights -= lr * numpy.dot(inputlayer, l1_delta)

            for num in z_delta:
                bias2 -= lr * num

            for num in l1_delta:
                bias -= lr * num
            print(iterration)
            if(iterration < 14):
                iterration += 1
            else:
                iterration = 0
                # Save Synapse / Model into Pickle
                pickle_out = open("pickle/syn0.pickle", "wb")
                pickle.dump(weights, pickle_out)
                pickle_out = open("pickle/syn1.pickle", "wb")
                pickle.dump(weights2, pickle_out)
                pickle_out = open("pickle/bias.pickle", "wb")
                pickle.dump(bias, pickle_out)
                pickle_out = open("pickle/bias2.pickle", "wb")
                pickle.dump(bias2, pickle_out)
                pickle_out.close()
                print("model updated")