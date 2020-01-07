require 'neural_network'

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
end
