import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # compute the loss and the gradient                                           
  num_classes = W.shape[1]                                                      
  num_train = X.shape[0]                                                        
  loss = 0.0                                                                    

  for i in xrange(num_train):                                                   
    # cross-entropy loss
    # note: technically it doesn’t make sense to talk about the “softmax loss”, 
    # since softmax is just the squashing function
    scores = X[i].dot(W)

    # to fix possible Numeric stability issue
    # http://cs231n.github.io/linear-classify/ "Practical issues: Numeric stability"
    # first shift the values of f so that the highest number is 0
    scores -= np.max(scores)
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
 
    # gradient
    for j in xrange(num_classes):                                               
      sm_score = np.exp(scores[j]) / np.sum(np.exp(scores))
                                                                               
      if j == y[i]:
        dW[:, j] += (-1 + sm_score) * X[i]
      else:
        dW[:, j] += sm_score * X[i]
                                                                               
  # Right now the loss is a sum over all training examples, but we want it      
  # to be an average instead so we divide by num_train.                         
  loss /= num_train                                                             
                                                                                
  # Add regularization to the loss.                                             
  loss += reg * np.sum(W * W)  

  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]                                                        
  scores = np.dot(X, W)                                                         
                                                                               
  sm_score = np.exp(scores) / np.sum(np.exp(scores), axis=1)[..., np.newaxis]
  sm_score[range(num_train), y] -= 1

  dW = np.dot(X.T, sm_score)
  dW = dW / num_train + 2 * reg * W
                                                                                
  correct_class_scores = np.choose(y, scores.T)
  loss = -correct_class_scores + np.log(np.sum(np.exp(scores), axis=1))
  loss = np.sum(loss)

  loss = loss / num_train + reg * np.sum(W * W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

