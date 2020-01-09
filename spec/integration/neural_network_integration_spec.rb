require 'neural_network'

describe 'neural_network' do
  describe 'train' do
    describe 'with [2, 4, 2] weight parameters' do
      it 'trains and updates the weights' do
        neural_network = RubyNN::NeuralNetwork.new([2, 4, 2], 1)

        weight_matrix = [
          [[0.5, 0.5], [0.7, 0.1], [0.8, 0.9], [0.5, 0.7]],
          [[0.2, 0.4, 0.1, 0.2], [0.2, 0.4, 0.1, 0.9]]
        ]
        neural_network.set_weights(weight_matrix)
        input = [2, 3]
        target_output = [4, 6]

        expected = [
          [
            [1.174, 1.174],
            [2.048, 1.448],
            [1.137, 1.237],
            [2.2939999999999996, 2.4939999999999998]
          ],
          [
            [1.9699999999999995, 2.1699999999999995, 1.8699999999999997, 1.9699999999999995],
            [1.7999999999999996, 1.9999999999999996, 1.6999999999999997, 2.4999999999999996]
          ]
        ]

        neural_network.train(input, target_output)
        expect(neural_network.get_weights).to eq expected
      end
    end

    describe 'with [4, 6, 5, 2] weight parameters' do
      it 'trains and updates the weights' do
        neural_network = RubyNN::NeuralNetwork.new([4, 6, 5, 2], 0.1)

        weight_matrix = [
          [
            [0.3, 0.5, 0.9, 0.1],
            [0.4, 0.2, 0.8, 0.6],
            [0.1, 0.8, 0.2, 0.5],
            [0.9, 0.1, 0.3, 0.4],
            [0.6, 0.8, 0.6, 0.9],
            [0.7, 0.6, 0.8, 0.2]
          ],
          [
            [0.7, 0.9, 0.8, 0.9, 0.3, 0.6],
            [0.1, 0.2, 0.6, 0.2, 0.8, 0.1],
            [0.5, 0.1, 0.7, 0.2, 0.8, 0.1],
            [0.5, 0.6, 0.8, 0.1, 0.7, 0.3],
            [0.1, 0.7, 0.1, 0.5, 0.7, 0.1]
          ],
          [
            [0.8, 0.4, 0.7, 0.7, 0.1],
            [0.8, 0.1, 0.8, 0.7, 0.5]
          ]
        ]
        neural_network.set_weights(weight_matrix)

        input = [2, 3, 2, 3]
        target_output = [40, 60]

        expected = [
          [
            [2.173175, 2.3731750000000003, 2.773175, 1.9731750000000003],
            [2.6039330000000005, 2.4039330000000008, 3.003933000000001, 2.8039330000000007],
            [2.628441000000001, 3.3284410000000006, 2.728441000000001, 3.028441000000001],
            [2.5879900000000005, 1.7879900000000004, 1.9879900000000004, 2.0879900000000005],
            [3.0687960000000007, 3.268796000000001, 3.0687960000000007, 3.3687960000000006],
            [1.8317890000000001, 1.731789, 1.9317890000000002, 1.3317890000000001]
          ],
          [
            [1.7872800000000002, 1.9872800000000002, 1.8872800000000003, 1.9872800000000002, 1.3872800000000003, 1.6872800000000003],
            [0.17986999999999997, 0.27986999999999995, 0.67987, 0.27986999999999995, 0.87987, 0.17986999999999997],
            [1.6059600000000003, 1.2059600000000004, 1.8059600000000002, 1.3059600000000002, 1.9059600000000003, 1.2059600000000004],
            [1.45137, 1.55137, 1.75137, 1.0513700000000001, 1.65137, 1.25137],
            [0.8542700000000002, 1.4542700000000002, 0.8542700000000002, 1.2542700000000002, 1.4542700000000002, 0.8542700000000002]
          ],
          [
            [0.6131999999999999, 0.2131999999999998, 0.5131999999999998, 0.5131999999999998, 0.1],
            [2.3459000000000003, 1.6459000000000006, 2.3459000000000003, 2.2459000000000007, 2.0459000000000005]
          ]
        ]

        neural_network.train(input, target_output)
        expect(neural_network.get_weights).to eq expected
      end
    end

    describe 'when it is trained a lot' do
      it 'lowers the error rate' do
        neural_network = RubyNN::NeuralNetwork.new([4, 6, 5, 2], 0.0001)

        weight_matrix = [
          [
            [0.3, 0.5, 0.9, 0.1],
            [0.4, 0.2, 0.8, 0.6],
            [0.1, 0.8, 0.2, 0.5],
            [0.9, 0.1, 0.3, 0.4],
            [0.6, 0.8, 0.6, 0.9],
            [0.7, 0.6, 0.8, 0.2]
          ],
          [
            [0.7, 0.9, 0.8, 0.9, 0.3, 0.6],
            [0.1, 0.2, 0.6, 0.2, 0.8, 0.1],
            [0.5, 0.1, 0.7, 0.2, 0.8, 0.1],
            [0.5, 0.6, 0.8, 0.1, 0.7, 0.3],
            [0.1, 0.7, 0.1, 0.5, 0.7, 0.1]
          ],
          [
            [0.8, 0.4, 0.7, 0.7, 0.1],
            [0.8, 0.1, 0.8, 0.7, 0.5]
          ]
        ]
        neural_network.set_weights(weight_matrix)

        input = [2, 3, 3, 3]
        target_output = [4000, 6000]

        one_iteration = 35408093.22090001
        ten_thousand_iterations = 0.0
        neural_network.train(input, target_output)
        expect(neural_network.error).to eq one_iteration

        10000.times do |n|
          neural_network.train(input, target_output)
        end

        expect(neural_network.error.round(20)).to eq ten_thousand_iterations
      end
    end
  end
end
