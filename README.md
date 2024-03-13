# train_quaternion.py

## Parameters
These are the parameters that can be set in the `train_quaternion.py` file:

```python
# Parameters
numQOut = 3 # Number of quaternions in the output layer

```

First we want to fix the format in which the input and output vector are converted to quaternions.
To do this we will use the following two functions:

```python   
stateToQuaternion(state)
actionToQuaternion(action)
```
For example the first function will take as input a vector state of 9 elements and will return a
vector of 12 elements, corresponding to 3 quaternion elements. 

These function are then used in the `loadDataController` function (but not only there) to convert the input and output vectors to the right quaternion format. 

Then the controller is created:
```python
controller_relu = utils.setup_relu_quaternion((3, 2, 2, numQOut),  
                                                  params=None,
                                                  negative_slope=0.01,
                                                  bias=True,
                                                  dtype=dtype)
```


