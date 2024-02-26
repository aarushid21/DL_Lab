import torch
import numpy as np

#Illustrate the functions for reshaping, viewing, stacking, squeezing and unsqueezing of tensors

t = torch.tensor([[11, 12, 13], [5, 9, 15]])
print("Original shape of tensor is", t.shape)
print("New shape of tensor is", (t.reshape(3, 2)).shape)
print(t)
a = torch.tensor([[3, 33, 333], [5, 55, 555]])
print("Stacking tensor A on T produces:")
print(torch.cat((t, a)))
c = torch.zeros(2, 1, 3, 1)
print("Tensor C original shape: ", c.shape)
print("Squeezing tensor C produces shape:", torch.squeeze(c).shape)
print("Unsqueezing tensor C produces shape:", torch.unsqueeze(c, 3).shape)

#Illustrate the use of torch.permute()
d = torch.zeros(3, 2, 1, 4)
print(d)
print("The permuted shape is: ", torch.permute(d, (2, 1, 0, 3)).shape)

#Illustrate indexing in tensors
print("Indexing of tensor A")
print(a[1][1])
print(a[0:2][1])

#Show how numpy arrays are converted to tensors and back to numpy arrays
a = np.array([3, 2, 1, 9])
print("A is " + str(a) + " of type", type(a))
a_tensor = torch.from_numpy(a)
print("Now, A is " + str(a_tensor) + " of type", type(a_tensor))
a = a_tensor.numpy()
print("Now, A is " + str(a) + " of type", type(a))

#Create a random tensor with shape (7, 7)
rand = torch.rand((7, 7))
print(rand)

#Perform a matrix multiplication on the tensor with a random tensor of (1, 7)
rand_2 = torch.rand(1, 7)
rand_t = torch.transpose(rand_2, 0, 1)
res = torch.matmul(rand, rand_t)
print("The resultant matrix is ", res)

#Create two random tensors of shape (2, 3) and send them both to the GPU
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a, a.device)
a_gpu = a.to('cuda:0')
b_gpu = b.to('cuda:0')
print(a_gpu, a_gpu.device )
print(b_gpu)

#Perform a matrix multiplication on the tensors
b_t = torch.transpose(b, 0, 1)
res = torch.matmul(a, b_t)
print("The resultant matrix is", res)

#Find the maximum and minimum values of the output
print("The maximum value is " + str(torch.max(res)) + " and the minimum value is", torch.min(res))

#Find the index of the maximum and minimum values of the output
print("The index of the maximum element is", ((res == torch.max(res)).nonzero(as_tuple=False)[0]))
print("The index of the minimum element is", ((res == torch.min(res)).nonzero(as_tuple=False)[0]))

#Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape
rand = torch.rand(1, 1, 1, 10)
rand_sq = torch.squeeze(rand)
print(rand_sq)
print(rand_sq.shape)
print("Setting seed to 7")
torch.manual_seed(7)
rand = torch.rand(1, 1, 1, 10)
rand_sq = torch.squeeze(rand)
print(rand_sq)
print(rand_sq.shape)
