import cv2, os, glob, re, time, numpy

start = time.time()
def splitstring(word):
    label = numpy.zeros(90)
    personId = int(word[2:4])
    label[personId-1] = 1
    return label

data = []
label = []
name = []
# read several image
for i in range(90):
    if(i == 79):
        continue
    print("read data : ",numpy.round(float(i)/89*100,2),'%')
    img_dir = "dataset/90subject-Cleaned/Subject" + str(i+1) # Enter Directory of all images 
    types = ('*.jpg','*.Jpg')
    files = []
    for f in types:
        files.extend(glob.glob(os.path.join(img_dir,f)))

    for f1 in files:
        image = cv2.imread(f1)
        base = os.path.basename(f1)
        base = os.path.splitext(base)
        degree = base[0][5:]
        if(degree == '0' or degree == '+15' or degree == '-15' or degree == '+30' or degree == '-30' or degree == '+45' or degree == '-45'):
            title = splitstring(base[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,(480,360))
            # fit into list
            data.append(image)
            label.append(title)
            name.append(base[0])

data = numpy.array(data)
label = numpy.array(label)
name = numpy.array(name)
end = time.time()
print("read data complete " , round(end-start,2) , "s, total : ", len(data))
