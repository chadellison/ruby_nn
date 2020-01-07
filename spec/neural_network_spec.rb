require 'ruby_nn'

describe 'neural_network' do
  it 'can be initialize' do
    neural_network = RubyNN::NeuralNetwork.new([1, 2, 3], 0.1)
    expect(neural_network.layer_parameters).to eq [1, 2, 3]
    expect(neural_network.alpha).to eq 0.1
  end

  describe 'initialize_weights' do
    it 'creates weights' do
      allow_any_instance_of(RubyNN::NeuralNetwork).to receive(:weight_counts)
        .and_return([10, 70, 20])

      neural_network = RubyNN::NeuralNetwork.new([1, 2, 3], 0.1)
      actual = neural_network.initialize_weights

      expect(actual.size).to eq 100
      expect(actual.all? { |w| w > 0 && w < 1 }).to be true
    end
  end

  describe 'offsets' do
    it 'returns an offset array based on the weight_counts' do
      allow_any_instance_of(RubyNN::NeuralNetwork).to receive(:weight_counts)
        .and_return([10, 70, 20])

      neural_network = RubyNN::NeuralNetwork.new([1, 2, 3], 0.1)
      expected = [0, 10, 80]

      actual = neural_network.offsets

      expect(actual).to eq expected
    end
  end

  describe 'set_weights' do
    it 'sets the weights' do
      neural_network = RubyNN::NeuralNetwork.new([1, 2, 3], 0.1)
      weights = [0.3, 0.4, 0.8, 0.4]

      actual = neural_network.set_weights(weights)

      expect(actual).to eq weights
    end
  end

  describe 'weight_counts' do
    it 'returns the weight counts based on the layer_parameters' do
      neural_network = RubyNN::NeuralNetwork.new([16, 20, 15, 10, 3], 0.1)
      expected = [320, 300, 150, 30]

      actual = neural_network.weight_counts

      expect(actual).to eq expected
    end
  end

  describe 'calculate_prediction' do
    it 'returns the result of the feed forward propagation' do
      neural_network = RubyNN::NeuralNetwork.new([16, 20, 15, 10, 3], 0.1)
      expected = [270000.0, 270000.0, 270000.0]
      weights = []
      800.times { weights << 1.0 }

      neural_network.set_weights(weights)
      input = [2, 1, 6, 4, 5, 3, 2, 3, 4, 10, 23, 12, 2, 4, 6, 3]
      actual = neural_network.calculate_prediction(input)

      expect(actual).to eq expected
    end
  end

  describe 'weighted_sum' do
    describe 'when the array length is different from the input length' do
      it 'raises an exception' do
        neural_network = RubyNN::NeuralNetwork.new([16, 20, 15, 10, 3], 0.1)

        expect { neural_network.weighted_sum([1], [1, 2]) }.to raise_error(RubyNN::NeuralNetworkError, 'arrays are not equal length')
      end
    end

    it 'performs a weighted sum operation' do
      neural_network = RubyNN::NeuralNetwork.new([16, 20, 15, 10, 3], 0.1)

      expect(neural_network.weighted_sum([3, 4], [4, 6])).to eq 36
    end
  end

  describe 'multiply_vector' do
    it 'calls weighted_sum with each weight set' do
      neural_network = RubyNN::NeuralNetwork.new([16, 20, 15, 10, 3], 0.1)
      input = [2, 3, 2]
      weight_matrix = [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]

      expected = [3.5, 3.5]
      expect(neural_network).to receive(:weighted_sum).with(input, [0.5, 0.5, 0.5])
      expect(neural_network).to receive(:weighted_sum).with(input, [0.3, 0.3, 0.3])

      neural_network.multiply_vector(input, weight_matrix)
    end

    it 'multiplies the input vector by the weight matrix' do
      neural_network = RubyNN::NeuralNetwork.new([16, 20, 15, 10, 3], 0.1)
      input = [2, 3, 2]
      weight_matrix = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

      expected = [3.5, 3.5]

      actual = neural_network.multiply_vector(input, weight_matrix)

      expect(actual).to eq expected
    end
  end

  describe 'find_weights' do
    it 'returns the correct weight matrix' do
      allow_any_instance_of(RubyNN::NeuralNetwork).to receive(:weight_counts)
        .and_return([2, 4, 4])

      allow_any_instance_of(RubyNN::NeuralNetwork).to receive(:offsets)
        .and_return([0, 2, 4])

      neural_network = RubyNN::NeuralNetwork.new([1, 2, 2, 2], 0.1)
      weights = [3, 3, 4, 4, 4, 4, 7, 7, 7, 7]
      neural_network.set_weights(weights)

      expected = [[4, 4], [4, 4]]

      actual = neural_network.find_weights(1)

      expect(actual).to eq expected
    end
  end

  describe 'train' do
    it 'calls calculate_prediction, create_deltas, and handle_weights' do
      neural_network = RubyNN::NeuralNetwork.new([1, 2, 2, 2], 0.1)

      input = [2, 4, 3]
      target_output = [5]

      expect(neural_network).to receive(:calculate_prediction)
        .with(input)

      expect(neural_network).to receive(:create_deltas)
        .with(target_output)

      expect(neural_network).to receive(:handle_weights)

      neural_network.train(input, target_output)
    end
  end

  describe 'create_deltas' do
    it 'calls find_deltas, multiply_vector, and back_propagation_multiplyer' do
      neural_network = RubyNN::NeuralNetwork.new([1, 2, 2, 2], 0.1)
      neural_network.set_weights([1, 1, 2, 1, 6, 1, 1, 4, 1, 2, 1, 8])
      neural_network.calculate_prediction([1])

      expect(neural_network).to receive(:find_deltas)
      expect(neural_network).to receive(:multiply_vector).at_least(2).times
      expect(neural_network).to receive(:back_propagation_multiplyer).at_least(2).times

      neural_network.create_deltas([2])
    end
  end
end
