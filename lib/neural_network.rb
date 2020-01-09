require 'neural_network_error'
require 'pry'
module RubyNN
  class NeuralNetwork
    attr_reader :layer_parameters, :alpha, :error

    def initialize(layer_parameters, alpha = 0.001)
      @alpha = alpha
      @weight_matrix = []
      @layer_parameters = layer_parameters
      @error = 0
    end

    def initialize_weights
      weights = []
      weight_counts.reduce(0, :+).times { weights << rand }
      layer_parameters[0..-2].each_with_index do |layer, i|
        @weight_matrix[i] = find_weights(i, weights)
      end
      @weight_matrix
    end

    def offsets
      if @offsets
        @offsets
      else
        @offsets = [0]
        weight_count_size = weight_counts.size

        weight_counts.each_with_index do |weight_count, i|
          if weight_count_size > i + 1
            @offsets << @offsets.last + weight_count
          end
        end
        @offsets
      end
    end

    def set_weights(weight_matrix)
      @weight_matrix = weight_matrix
    end

    def weight_counts
      if @weight_counts
        @weight_counts
      else
        @weight_counts = []
        layer_parameters.each_with_index do |count, i|
          if layer_parameters[i + 1]
            @weight_counts << (layer_parameters[i] * layer_parameters[i + 1])
          end
        end
        @weight_counts
      end
    end

    def calculate_prediction(input)
      predictions = []
      layer_parameters[0..-2].each_with_index do |layer, i|
        input_value = i == 0 ? input : predictions[i - 1]
        prediction_vector = multiply_vector(input_value, @weight_matrix[i])
        prediction_vector = leaky_relu(prediction_vector) if layer_parameters[0..-2][i + 1]
        predictions << prediction_vector
      end
      predictions
    end

    def weighted_sum(input, weights)
      total_weight = 0
      raise raise NeuralNetworkError, 'arrays are not equal length' if input.size != weights.size
      input.size.times do |index|
        total_weight += input[index] * weights[index]
      end
      total_weight
    end

    def multiply_vector(input, weight_matrix)
      predictions = []
      weight_matrix.size.times do |index|
        predictions[index] = weighted_sum(input, weight_matrix[index])
      end
      predictions
    end

    def find_weights(i, weights)
      weight_amount, offset, slice_value = weight_counts[i], offsets[i], layer_parameters[i]
      weights[(offset)...(offset + weight_amount)].each_slice(slice_value).to_a
    end

    def train(input, target_output)
      predictions = calculate_prediction(input)
      back_propagate(predictions, target_output)
    end

    def back_propagate(predictions, target_output)
      reversed_weight_matrix = @weight_matrix.reverse
      last_weighted = []
      predictions.reverse.each_with_index do |prediction, i|
        delta_set = find_deltas(prediction, target_output) if i == 0
        if i != 0
          delta_set = back_propagation_multiplyer(last_weighted, relu_derivative(prediction))
        end
        weighted = multiply_vector(delta_set, reversed_weight_matrix[i].transpose)
        last_weighted = weighted
        update_weights(delta_set, i)
      end
    end

    def save_weights(filename)
      File.open(filename, "w") do |f|
        f.write(@weight_matrix.to_json)
      end
      puts 'saved weights to ' + filename
    end

    def find_deltas(predictions, outcomes)
      deltas = []
      predictions.size.times do |index|
        delta = predictions[index] - outcomes[index]
        deltas[index] = delta
        @error = delta ** 2
      end

      deltas
    end

    def update_weights(weighted_deltas, i)
      reversed_weight_matrix = @weight_matrix.reverse
      @weight_matrix.reverse[i].size.times do |index|
        @weight_matrix.reverse[i][index].size.times do |count|
          weight = @weight_matrix.reverse[i][index][count]
          adjusted_value = (weight - (@alpha * weighted_deltas[index]))
          @weight_matrix.reverse[i][index][count] = adjusted_value if adjusted_value > 0
        end
      end
    end

    def calculate_deltas(input, deltas)
      weighted_deltas = []
      deltas.each { weighted_deltas.push([]) }

      deltas.size.times do |index|
        input.size.times do |count|
          weighted_deltas[index][count] = input[count] * deltas[index]
        end
      end

      weighted_deltas
    end

    def leaky_relu(input)
      input.map { |value| value > 0 ? value : 0.0001 }
    end

    def relu_derivative(output)
      output.map { |value| value > 0 ? 1 : 0.0001 }
    end

    def calculate_outcomes(abstraction)
      first = 0.0
      second = 0.0
      third = 0.0
      abstraction.setups.each do |setup|
        white_wins = setup.outcomes[:white_wins].to_f
        black_wins = setup.outcomes[:black_wins].to_f
        draws = setup.outcomes[:draws].to_f

        if setup.position_signature[-1] == 'w'
          first += white_wins
          second += black_wins
        else
          second += black_wins
          first += white_wins
        end

        third = draws
      end

      [first, second, third]
    end

    def back_propagation_multiplyer(v1, v2)
      v1.zip(v2).map { |set| set[0] * set[1] }
    end

    def softmax(vector)
      sum = vector.sum.to_f
      vector.map do |value|
        if value == 0
          0
        else
          value / sum
        end
      end
    end

    def get_weights
      @weight_matrix
    end
  end
end
