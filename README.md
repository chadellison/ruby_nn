# README

## A customizable neural network built in ruby!

### Instantiate a Neural Network
- The neural network takes two arguments: 1) an array of layer parameters and an optional learning rate (alpha)
- ```RubyNN::NeuralNetwork.new([12, 20, 22, 3], 0.01)```

- the learning rate defaults to 00.1 if no 2nd parameter is given
- ```RubyNN::NeuralNetwork.new([12, 20, 22, 3])```
### The layer parameters
- The first element in the array is the number of nodes in the input layer and the last element in the array is the number of nodes in the output layer. These as well as the number of nodes for the hidden layers are configurable. For example, the following will initialize a network with 4 input nodes, one hidden layer with 8 nodes, and one output layer with 6 nodes: ```RubyNN::NeuralNetwork.new([4, 8, 6])
=> #<RubyNN::NeuralNetwork:0x00007faa712f98d8
 @alpha=0.001,
 @deltas=[],
 @layer_parameters=[4, 8, 6],
 @predictions=[],
 @weight_matrix=[],
 @weights=[]>
 @error=0
 ```

- Generating random weights
```neural_network = RubyNN::NeuralNetwork.new([4, 8, 6])
neural_network.initialize_weights
```
This will load the appropriate number of weights for the given layer parameters =>
```@weights=
  [
    0.27467491765174346,
     0.12210417074805302,
     0.8003157818002387,
     0.6389939906739756,
     0.08775650196076201,
     0.7875689188665604
   ...]
```
The weights can also be set on the Network
```
neural_network.set_weights([0.27467491765174346, ...])
```

- To feed data through the network:
  ```
  input = [3.5, 2.2, 0.3, 1.8]
  neural_network.calculate_prediction(input)
  => [26.946181168022996, 25.464420767593115, 23.07104915919065, 17.785122540081424, 30.773243913480776, 16.450739844671077]
  ```

- To train the network:
```
input = [3.5, 2.2, 0.3, 1.8]
target_output = [3, 1, 1.2, 0.3, 0.2, 2.1]
neural_network.train(input, target_output)
```

note: this would be done in a loop:
```
inputs = [[3.5, 2.2, 0.3, 1.8], [1.5, 8.2, 2.3, 1.8], ...]
target_outputs = [[3, 1, 1.2, 0.3, 0.2, 2.1], [2, 1.2, 1.1, 2.3, 4.2, 1.1], ...]
inputs.each_with_index do |input, i|
  neural_network.train(input, target_outputs[i])
  puts neural_network.error.to_s if % 100 == 0 # => to see the error rate
end
```

- Save the weights after training and then set on a new instance of the neural_network
```
neural_network.save_weights('your_file.json')
```
this will write a json file of the weights array
```
new_instance = RubyNN::NeuralNetwork.new([4, 8, 6])
new_instance.set_weights(JSON.parse(your_file.json))
input = [2, 3, 1.3, 7]
new_instance.calculate_prediction(input)
```

- The activation function for all layers is leaky_relu (There are plans to make this configurable so an activation function can be specified)

- Ruby version: 2.6.4

- The test suite can be run with:
  - `rspec`

- Feel free to contribute by submitting a PR
