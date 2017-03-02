import pickle
import cv2

# deserialize
with open('out.pkl', 'rb') as f:
  data = pickle.load(f)
print data[1].shape
print data[3].shape
print data[5].shape
print data[6]

# img = cv2.imread("sample.jpg")
# with open("datasets.pkl", "wb") as f:
#     pickle.dump(img,f)


# # test data
# data = ['this','is','a','test','.']
# appendix = ['this','is','appendix','.']

# print('raw data:\n'+str(data))
# # serialize
# with open('pickle_test.dump', 'wb') as f:
#   pickle.dump(data, f)

# # deserialize
# with open('pickle_test.dump', 'rb') as f:
#   data = pickle.load(f)

# print('pickle data:\n'+str(data))

# # try append
# with open('pickle_test.dump', 'ab') as f:
#   pickle.dump(appendix, f)

# with open('pickle_test.dump', 'rb') as f:
#   data = pickle.load(f)

# print('appended pickle data:\n'+str(data))