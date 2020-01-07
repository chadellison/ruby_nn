require 'neural_network_error'

module RubyNN
  class NeuralNetwork
    # ALPHA = 0.001
    # WEIGHT_COUNTS = [320, 300, 150, 30]
    # OFFSETS = [0, 320, 620, 770]
    # VECTOR_COUNTS = [16, 20, 15, 10, 3]
    #
    # include CacheLogic
    #
    # attr_accessor :layer_one_predictions, :layer_two_predictions,
    #   :layer_three_predictions, :layer_four_predictions, :layer_one_deltas,
    #   :layer_two_deltas, :layer_three_deltas, :layer_four_deltas,
    #
    # def initialize
    #   @layer_one_predictions = []
    #   @layer_two_predictions = []
    #   @layer_three_predictions = []
    #   @layer_four_predictions = []
    #   @layer_one_deltas = []
    #   @layer_two_deltas = []
    #   @layer_three_deltas = []
    #   @layer_four_deltas = []
    # end

    attr_reader :layer_parameters, :alpha

    def initialize(layer_parameters, alpha = 0.001)
      @predictions = []
      @alpha = alpha
      @weights = []
      @weight_matrix = []
      @deltas = []
      @layer_parameters = layer_parameters
    end

    def initialize_weights
      weight_counts.reduce(0, :+).times { @weights << rand }
      @weights
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

    def set_weights(weights)
      @weights = weights
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

    # def layer_one_weights
    #   @layer_one_weights ||= find_weights(0)
    # end
    #
    # def layer_two_weights
    #   @layer_two_weights ||= find_weights(1)
    # end
    #
    # def layer_three_weights
    #   @layer_three_weights ||= find_weights(2)
    # end
    #
    # def layer_four_weights
    #   @layer_four_weights ||= find_weights(3)
    # end

    def calculate_prediction(input)
      # input = normalize_values(abstraction)
      # @layer_one_predictions = leaky_relu(multiply_vector(input, layer_one_weights))
      # @layer_two_predictions = leaky_relu(multiply_vector(layer_one_predictions, layer_two_weights))
      # @layer_three_predictions = leaky_relu(multiply_vector(layer_two_predictions, layer_three_weights))
      # @layer_four_predictions = (multiply_vector(layer_three_predictions, layer_four_weights))

      layer_parameters[0..-2].each_with_index do |layer, i|
        input_value = i == 0 ? input : @predictions[i - 1]
        prediction_vector = multiply_vector(input_value, find_weights(i))
        prediction_vector = leaky_relu(prediction_vector) if layer_parameters[0..-2][i + 1]
        @predictions << prediction_vector
      end
      @predictions.last
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

    def find_weights(i)
      weight_amount, offset, slice_value = weight_counts[i], offsets[i], layer_parameters[i]
      # range = ((offset + 1)..(offset + weight_amount))
      # weights = Weight.where(weight_count: range).order(:weight_count).map do |weight|
      #   weight.value.to_f
      # end
      weight_slice = @weights[(offset)...(offset + weight_amount)].each_slice(slice_value).to_a
      @weight_matrix << weight_slice
      weight_slice
    end

    def train(input, target_output)
      # input = normalize_values(abstraction)

      calculate_prediction(input)
      # outcomes = softmax(calculate_outcomes(abstraction))

      create_deltas(target_output)
      handle_weights
      # update_weights(layer_four_deltas, layer_four_weights)
      # update_weights(layer_three_deltas, layer_three_weights)
      # update_weights(layer_two_deltas, layer_two_weights)
      # update_weights(layer_one_deltas, layer_one_weights)
    end

    def create_deltas(outcomes)
      @deltas = []
      reversed_weight_matrix = @weight_matrix.reverse
      @predictions.reverse.each_with_index do |prediction, i|
        if i == 0
          @deltas << find_deltas(prediction, outcomes)
        else
          weighted = multiply_vector(@deltas.last, reversed_weight_matrix[i].transpose)
          @deltas << back_propagation_multiplyer(weighted, relu_derivative(prediction))
        end
      end


      # @layer_four_deltas = find_deltas(layer_four_predictions, outcomes)
      # l_4_weighted = multiply_vector(layer_four_deltas, layer_four_weights.transpose)
      # @layer_three_deltas = back_propagation_multiplyer(l_4_weighted, relu_derivative(layer_three_predictions))
      # l_3_weighted = multiply_vector(layer_three_deltas, layer_three_weights.transpose)
      # @layer_two_deltas = back_propagation_multiplyer(l_3_weighted, relu_derivative(layer_two_predictions))
      # l_2_weighted = multiply_vector(layer_two_deltas, layer_two_weights.transpose)
      # @layer_one_deltas = back_propagation_multiplyer(l_2_weighted, relu_derivative(layer_one_predictions))
    end

    # def save_weights
    #   all_weight_values = layer_one_weights.flatten +
    #                       layer_two_weights.flatten +
    #                       layer_three_weights.flatten +
    #                       layer_four_weights.flatten
    #
    #   Weight.order(:weight_count).each_with_index do |weight, index|
    #     weight.update(value: all_weight_values[index].to_s)
    #   end
    # end

    def find_deltas(predictions, outcomes)
      deltas = []
      predictions.size.times do |index|
        delta = predictions[index] - outcomes[index]
        deltas[index] = delta
        error = delta ** 2
        update_error_rate(error)
      end

      deltas
    end

    def handle_weights
      reversed_weight_matrix = @weight_matrix.reverse
      @deltas.each_with_index do |delta_set, i|
        update_weights(delta_set, reversed_weight_matrix[i])
      end
    end

    def update_weights(weighted_deltas, weight_matrix)
      weight_matrix.size.times do |index|
        weight_matrix[index].size.times do |count|
          weight = weight_matrix[index][count]
          adjusted_value = (weight - (@alpha * weighted_deltas[index]))
          weight_matrix[index][count] = adjusted_value if adjusted_value > 0
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

    def update_error_rate(error)
      error_object = JSON.parse(get_from_cache('error_rate')).symbolize_keys
      error_object[:count] += 1
      error_object[:error] += error
      add_to_cache('error_rate', error_object)
    end

    # def normalize_values(abstraction)
    #   abstraction.pattern.split('-').map { |value| value.to_f }
    # end

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
  end
end
