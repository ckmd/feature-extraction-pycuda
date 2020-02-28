from pykafka import KafkaClient
import json, pandas, pickle, numpy, time

# print("Initializing ... ")
topicname = 'train5'
client = KafkaClient(hosts='localhost:9092')
consumer = client.topics[topicname].get_simple_consumer()

def sigmoid(x):
    return 1/(1+numpy.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# load trained model
# weights = pickle.load(open("model/2syn0.pickle", "rb"))
# weights2 = pickle.load(open("model/2syn1.pickle", "rb"))
# bias = pickle.load(open("model/2bias.pickle", "rb"))
# bias2 = pickle.load(open("model/2bias2.pickle", "rb"))

# Defining Neural Network Synapse for the first time
numpy.random.seed(0)
weights = 2 * numpy.random.rand(2176,900) - 1
weights2 = 2 * numpy.random.rand(900,5) - 1
bias = 2 * numpy.random.rand(1,900) - 1
bias2 = 2 * numpy.random.rand(1,5) - 1
lr = 0.05
iterration = 1
datake = 1
accuracy = 0
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
            label = {int(k):float(v) for k,v in valjson['label'].items()}
            # restructure dict into array
            inputlayer = numpy.array([features.values()])
            labellayer = numpy.array([label.values()])
            # forward propagation
            l1 = sigmoid(numpy.dot(inputlayer, weights) + bias)
            z = sigmoid(numpy.dot(l1, weights2) + bias2)
            # backpropagation step 1
            error = z - labellayer
            # print(z)
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

            end = time.time()
            print('pred : ',numpy.argmax(z),'truth : ',numpy.argmax(labellayer))
            if(numpy.argmax(z) == numpy.argmax(labellayer)):
                accuracy += 1
            print('data ke : ',datake, 'acc : ',float(accuracy)/datake*100,'time : ',end-start)
            datake += 1

            if(iterration < 70):
                iterration += 1
            else:
                iterration = 1
                print("updating model ...")
                # Save Synapse / Model into Pickle
                pickle_out = open("model/5labelsyn0.pickle", "wb")
                pickle.dump(weights, pickle_out)
                pickle_out = open("model/5labelsyn1.pickle", "wb")
                pickle.dump(weights2, pickle_out)
                pickle_out = open("model/5labelbias.pickle", "wb")
                pickle.dump(bias, pickle_out)
                pickle_out = open("model/5labelbias2.pickle", "wb")
                pickle.dump(bias2, pickle_out)
                pickle_out.close()
                print("model updated")