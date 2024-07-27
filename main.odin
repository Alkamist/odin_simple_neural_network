package main

import "base:runtime"
import "core:c"
import "core:os"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:strconv"
import "core:encoding/csv"
import "core:encoding/json"
import "core:image"
import "core:image/png"

INPUT_SIZE :: 784
HIDDEN_SIZE :: 100
OUTPUT_SIZE :: 10

TRAINING_EPOCHS :: 10

main :: proc() {
    train()
    // load_and_validate()
    // predict_digit()
}

train :: proc() {
    defer fmt.println("Training Complete")
    defer free_all(context.temp_allocator)

    training_set: Data_Set
    data_set_init(&training_set, 60000, INPUT_SIZE, OUTPUT_SIZE, context.temp_allocator)
    if !data_set_load_csv(&training_set, "mnist_train.csv") {
        fmt.eprintln("Failed to load mnist_train.csv")
    }

    validation_set: Data_Set
    data_set_init(&validation_set, 10000, INPUT_SIZE, OUTPUT_SIZE, context.temp_allocator)
    if !data_set_load_csv(&validation_set, "mnist_test.csv") {
        fmt.eprintln("Failed to load mnist_test.csv")
    }

    mlp: MLP
    mlp_init(&mlp, {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE})
    defer mlp_destroy(&mlp)

    mlp_randomize_he(&mlp)

    best_score := 0

    for i in 0 ..< TRAINING_EPOCHS {
        data_set_shuffle(&training_set)
        mlp_optimize_adam(&mlp, &training_set,
            batch_size = 16,
            learning_rate = 0.001,
            regularization_lambda = 0.0001,
            dropout_rate = 0.5,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8,
        )

        correct_answer_count := mlp_validate(&mlp, &validation_set)
        fmt.printfln("Accuracy is %v%%", 100.0 * f32(correct_answer_count) / f32(validation_set.size))

        if correct_answer_count > best_score {
            best_score = correct_answer_count

            checkpoint: MLP_Checkpoint
            mlp_checkpoint_init(&checkpoint, &mlp)
            mlp_save_checkpoint(&mlp, &checkpoint)
            mlp_checkpoint_save_to_json_file(&checkpoint, "model.json")
        }
    }
}

load_and_validate :: proc() {
    defer free_all(context.temp_allocator)

    validation_set: Data_Set
    data_set_init(&validation_set, 10000, INPUT_SIZE, OUTPUT_SIZE, context.temp_allocator)
    if !data_set_load_csv(&validation_set, "mnist_test.csv") {
        fmt.eprintln("Failed to load mnist_test.csv")
    }

    mlp: MLP
    mlp_init(&mlp, {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE})
    defer mlp_destroy(&mlp)

    checkpoint: MLP_Checkpoint
    mlp_checkpoint_init(&checkpoint, &mlp)
    mlp_checkpoint_load_from_json_file(&checkpoint, "model.json")

    mlp_load_checkpoint(&mlp, &checkpoint)

    correct_answer_count := mlp_validate(&mlp, &validation_set)
    fmt.printfln("Accuracy is %v%%", 100.0 * f32(correct_answer_count) / f32(validation_set.size))
}

predict_digit :: proc() {
    defer free_all(context.temp_allocator)

    mlp: MLP
    mlp_init(&mlp, {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE})
    defer mlp_destroy(&mlp)

    checkpoint: MLP_Checkpoint
    mlp_checkpoint_init(&checkpoint, &mlp)
    mlp_checkpoint_load_from_json_file(&checkpoint, "model.json")

    mlp_load_checkpoint(&mlp, &checkpoint)

    digit_image, err := png.load_from_file("digit.png")
    if err != nil {
        fmt.eprintln("Failed to load digit.png")
    }

    pixels := mem.slice_data_cast([]image.RGBA_Pixel, digit_image.pixels.buf[:])

    input: [INPUT_SIZE]f32
    for i in 0 ..< INPUT_SIZE {
        p := pixels[i]
        average_rgb := (f32(p.r) + f32(p.g) + f32(p.b)) / (255.0 * 3.0)
        input[i] = average_rgb * f32(p.a) / 255.0
    }

    // See what the model thinks the number is.
    mlp_forward(&mlp, input[:])
    answer := slice.max_index(mlp_output(&mlp)[:])

    fmt.printfln("The digit is %v", answer)

    // Just to get the console to pause.
    buffer: [256]byte
    os.read(os.stdin, buffer[:])
}

//==========================================================================
// Multi-Layer Perceptron
//==========================================================================

MLP_Checkpoint :: struct {
    weights: []f32,
    biases: []f32,
}

MLP :: struct {
    weight_layer_offsets: []int,
    weights: []f32,
    weight_gradients: []f32,

    bias_layer_offsets: []int,
    biases: []f32,
    bias_gradients: []f32,

    activation_offsets: []int,
    activations: []f32,
    zs: []f32,
    deltas: []f32,

    neuron_counts: []int,

    adam_optimizer: struct {
        timestep: int,
        m_weights: []f32,
        v_weights: []f32,
        m_biases: []f32,
        v_biases: []f32,
    }
}

mlp_init :: proc(mlp: ^MLP, neuron_counts: []int, allocator := context.allocator) {
    assert(len(neuron_counts) >= 2)

    mlp.neuron_counts = make([]int, len(neuron_counts), allocator)
    copy(mlp.neuron_counts, neuron_counts)

    mlp.weight_layer_offsets = make([]int, len(neuron_counts) - 1, allocator)
    mlp.bias_layer_offsets = make([]int, len(neuron_counts) - 1, allocator)
    mlp.activation_offsets = make([]int, len(neuron_counts), allocator)

    bias_count := 0
    weight_count := 0
    activation_count := neuron_counts[0]
    for i in 1 ..< len(neuron_counts) {
        mlp.weight_layer_offsets[i - 1] = weight_count
        mlp.bias_layer_offsets[i - 1] = bias_count
        mlp.activation_offsets[i] = activation_count
        weight_count += neuron_counts[i] * neuron_counts[i - 1]
        bias_count += neuron_counts[i]
        activation_count += neuron_counts[i]
    }

    mlp.biases = make([]f32, bias_count, allocator)
    mlp.bias_gradients = make([]f32, bias_count, allocator)
    mlp.weights = make([]f32, weight_count, allocator)
    mlp.weight_gradients = make([]f32, weight_count, allocator)
    mlp.activations = make([]f32, activation_count, allocator)
    mlp.zs = make([]f32, activation_count, allocator)
    mlp.deltas = make([]f32, activation_count, allocator)

    mlp.adam_optimizer.m_weights = make([]f32, len(mlp.weights), allocator)
    mlp.adam_optimizer.v_weights = make([]f32, len(mlp.weights), allocator)
    mlp.adam_optimizer.m_biases = make([]f32, len(mlp.biases), allocator)
    mlp.adam_optimizer.v_biases = make([]f32, len(mlp.biases), allocator)
}

mlp_destroy :: proc(mlp: ^MLP) {
    delete(mlp.weight_layer_offsets)
    delete(mlp.weights)
    delete(mlp.weight_gradients)
    delete(mlp.bias_layer_offsets)
    delete(mlp.biases)
    delete(mlp.bias_gradients)
    delete(mlp.activation_offsets)
    delete(mlp.activations)
    delete(mlp.zs)
    delete(mlp.deltas)
    delete(mlp.neuron_counts)

    delete(mlp.adam_optimizer.m_weights)
    delete(mlp.adam_optimizer.v_weights)
    delete(mlp.adam_optimizer.m_biases)
    delete(mlp.adam_optimizer.v_biases)
}

mlp_randomize_xavier :: proc(mlp: ^MLP) {
    // Xavier initialization
    for i in 1 ..< len(mlp.neuron_counts) {
        neuron_count := mlp.neuron_counts[i]
        previous_neuron_count := mlp.neuron_counts[i - 1]

        scale := 1.0 / math.sqrt_f32(f32(neuron_count * previous_neuron_count))

        for neuron_index in 0 ..< neuron_count {
            for previous_neuron_index in 0 ..< previous_neuron_count {
                weight_index := mlp_weight_index(mlp, i, previous_neuron_index, neuron_index)
                mlp.weights[weight_index] = scale * rand.float32_normal(0, 1)
            }
        }
    }
    for i in 0 ..< len(mlp.biases) {
        mlp.biases[i] = rand.float32_normal(0, 1)
    }
}

mlp_randomize_he :: proc(mlp: ^MLP) {
    // He initialization
    for i in 1 ..< len(mlp.neuron_counts) {
        neuron_count := mlp.neuron_counts[i]
        previous_neuron_count := mlp.neuron_counts[i - 1]

        scale := math.sqrt(2.0 / f32(previous_neuron_count))

        for neuron_index in 0 ..< neuron_count {
            for previous_neuron_index in 0 ..< previous_neuron_count {
                weight_index := mlp_weight_index(mlp, i, previous_neuron_index, neuron_index)
                mlp.weights[weight_index] = scale * rand.float32_normal(0, 1)
            }
        }
    }
    for i in 0 ..< len(mlp.biases) {
        mlp.biases[i] = rand.float32_normal(0, 1)
    }
}

mlp_activation_index :: proc(mlp: ^MLP, row, index: int) -> int {
    return mlp.activation_offsets[row] + index
}

mlp_weight_index :: proc(mlp: ^MLP, layer, from_neuron, to_neuron: int) -> int {
    return mlp.weight_layer_offsets[layer - 1] + mlp.neuron_counts[layer - 1] * to_neuron + from_neuron
}

mlp_bias_index :: proc(mlp: ^MLP, layer, neuron: int) -> int {
    return mlp.bias_layer_offsets[layer - 1] + neuron
}

mlp_output :: proc(mlp: ^MLP) -> []f32 {
    output_size := mlp.neuron_counts[len(mlp.neuron_counts) - 1]
    output_start := len(mlp.activations) - output_size
    return mlp.activations[output_start:][:output_size]
}

mlp_forward :: proc(mlp: ^MLP, input: []f32, dropout_rate := f32(0)) {
    for i in 0 ..< mlp.neuron_counts[0] {
        mlp.activations[i] = input[i]
    }

    for i in 1 ..< len(mlp.neuron_counts) {
        neuron_count := mlp.neuron_counts[i]
        previous_neuron_count := mlp.neuron_counts[i - 1]

        for neuron_index in 0 ..< neuron_count {
            bias_index := mlp_bias_index(mlp, i, neuron_index)
            activation_value := mlp.biases[bias_index]

            for previous_neuron_index in 0 ..< previous_neuron_count {
                weight_index := mlp_weight_index(mlp, i, previous_neuron_index, neuron_index)
                previous_activation_index := mlp_activation_index(mlp, i - 1, previous_neuron_index)
                activation_value += mlp.weights[weight_index] * mlp.activations[previous_activation_index]
            }

            activation_index := mlp_activation_index(mlp, i, neuron_index)
            mlp.zs[activation_index] = activation_value
            // mlp.activations[activation_index] = sigmoid(activation_value)

            if i == len(mlp.neuron_counts) - 1 {
                mlp.activations[activation_index] = activation_value
            } else {
                mlp.activations[activation_index] = leaky_relu(activation_value)
            }

            if dropout_rate > 0 && rand.float32() < dropout_rate {
                mlp.activations[i] = 0
            } else {
                mlp.activations[i] /= (1 - dropout_rate)
            }
        }
    }

    output_size := mlp.neuron_counts[len(mlp.neuron_counts) - 1]
    output_start := len(mlp.activations) - output_size
    softmax(mlp.activations[output_start:][:output_size])
    softmax(mlp.zs[output_start:][:output_size])
}

mlp_accumulate_gradients :: proc(mlp: ^MLP, input, target: []f32, dropout_rate: f32) {
    mlp_forward(mlp, input, dropout_rate)

    {
        layer := len(mlp.neuron_counts) - 1

        neuron_count := mlp.neuron_counts[layer]
        previous_neuron_count := mlp.neuron_counts[layer - 1]

        for neuron in 0 ..< neuron_count {
            activation_index := mlp_activation_index(mlp, layer, neuron)
            // Cross entropy loss
            delta := mlp.activations[activation_index] - target[neuron]
            // delta := (mlp.activations[activation_index] - target[neuron]) * sigmoid_derivative(mlp.zs[activation_index])
            mlp.deltas[activation_index] = delta

            bias_index := mlp_bias_index(mlp, layer, neuron)
            mlp.bias_gradients[bias_index] += delta

            for previous_neuron in 0 ..< previous_neuron_count {
                previous_activation_index := mlp_activation_index(mlp, layer - 1, previous_neuron)
                weight_index := mlp_weight_index(mlp, layer, previous_neuron, neuron)
                mlp.weight_gradients[weight_index] += delta * mlp.activations[previous_activation_index]
            }
        }
    }

    for layer := len(mlp.neuron_counts) - 2; layer > 0; layer -= 1 {
        neuron_count := mlp.neuron_counts[layer]
        previous_neuron_count := mlp.neuron_counts[layer - 1]
        next_neuron_count := mlp.neuron_counts[layer + 1]

        for neuron in 0 ..< neuron_count {
            d_cost_o := f32(0)
            for next_neuron in 0 ..< next_neuron_count {
                weight_index := mlp_weight_index(mlp, layer + 1, neuron, next_neuron)
                next_activation_index := mlp_activation_index(mlp, layer + 1, next_neuron)
                d_cost_o += mlp.deltas[next_activation_index] * mlp.weights[weight_index]
            }

            activation_index := mlp_activation_index(mlp, layer, neuron)
            // delta := d_cost_o * sigmoid_derivative(mlp.zs[activation_index])
            delta := d_cost_o * leaky_relu_derivative(mlp.zs[activation_index])
            mlp.deltas[activation_index] = delta

            bias_index := mlp_bias_index(mlp, layer, neuron)
            mlp.bias_gradients[bias_index] += delta

            for previous_neuron in 0 ..< previous_neuron_count {
                previous_activation_index := mlp_activation_index(mlp, layer - 1, previous_neuron)
                weight_index := mlp_weight_index(mlp, layer, previous_neuron, neuron)
                mlp.weight_gradients[weight_index] += delta * mlp.activations[previous_activation_index]
            }
        }
    }
}

mlp_clear_gradients :: proc(mlp: ^MLP) {
    for i in 0 ..< len(mlp.weight_gradients) {
        mlp.weight_gradients[i] = 0
    }
    for i in 0 ..< len(mlp.bias_gradients) {
        mlp.bias_gradients[i] = 0
    }
}

mlp_optimize_sgd :: proc(
    mlp: ^MLP,
    data_set: ^Data_Set,
    batch_size: int = 16,
    learning_rate: f32 = 0.001,
    regularization_lambda: f32 = 0.0001,
    dropout_rate: f32 = 0.5,
) {
    mlp_clear_gradients(mlp)

    for i in 0 ..< data_set.size {
        input, output := data_set_sample(data_set, i)
        mlp_accumulate_gradients(mlp, input, output, dropout_rate)

        if i % batch_size == 0 {
            for i in 0 ..< len(mlp.weights) {
                grad := mlp.weight_gradients[i] / f32(batch_size) + regularization_lambda * mlp.weights[i]
                mlp.weights[i] -= learning_rate * grad
            }
            for i in 0 ..< len(mlp.biases) {
                grad := mlp.bias_gradients[i] / f32(batch_size)
                mlp.biases[i] -= grad * learning_rate
            }

            mlp_clear_gradients(mlp)
        }
    }
}

mlp_reset_adam :: proc(mlp: ^MLP) {
    mlp.adam_optimizer.timestep = 0
    for i in 0 ..< len(mlp.weights) {
        mlp.adam_optimizer.m_weights[i] = 0
        mlp.adam_optimizer.v_weights[i] = 0
    }
    for i in 0 ..< len(mlp.biases) {
        mlp.adam_optimizer.m_biases[i] = 0
        mlp.adam_optimizer.v_biases[i] = 0
    }
}

mlp_optimize_adam :: proc(
    mlp: ^MLP,
    data_set: ^Data_Set,
    batch_size: int = 16,
    learning_rate: f32 = 0.001,
    regularization_lambda: f32 = 0.0001,
    dropout_rate: f32 = 0.5,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
) {
    opt := &mlp.adam_optimizer

    mlp_clear_gradients(mlp)

    for i in 0 ..< data_set.size {
        input, output := data_set_sample(data_set, i)
        mlp_accumulate_gradients(mlp, input, output, dropout_rate)

        if i % batch_size == 0 {
            opt.timestep += 1

            for i in 0 ..< len(mlp.weights) {
                grad := mlp.weight_gradients[i] / f32(batch_size) + regularization_lambda * mlp.weights[i]
                opt.m_weights[i] = beta1 * opt.m_weights[i] + (1 - beta1) * grad
                opt.v_weights[i] = beta2 * opt.v_weights[i] + (1 - beta2) * grad * grad
                m_corrected := opt.m_weights[i] / (1 - math.pow(beta1, f32(opt.timestep)))
                v_corrected := opt.v_weights[i] / (1 - math.pow(beta2, f32(opt.timestep)))
                mlp.weights[i] -= learning_rate * m_corrected / (math.sqrt(v_corrected) + epsilon)
            }

            for i in 0 ..< len(mlp.biases) {
                grad := mlp.bias_gradients[i] / f32(batch_size)
                opt.m_biases[i] = beta1 * opt.m_biases[i] + (1 - beta1) * grad
                opt.v_biases[i] = beta2 * opt.v_biases[i] + (1 - beta2) * grad * grad
                m_corrected := opt.m_biases[i] / (1 - math.pow(beta1, f32(opt.timestep)))
                v_corrected := opt.v_biases[i] / (1 - math.pow(beta2, f32(opt.timestep)))
                mlp.biases[i] -= learning_rate * m_corrected / (math.sqrt(v_corrected) + epsilon)
            }

            mlp_clear_gradients(mlp)
        }
    }
}

mlp_validate :: proc(mlp: ^MLP, data_set: ^Data_Set) -> (correct_answer_count: int) {
    for i in 0 ..< data_set.size {
        input, output := data_set_sample(data_set, i)
        mlp_forward(mlp, input)
        if slice.max_index(mlp_output(mlp)[:]) == slice.max_index(output) {
            correct_answer_count += 1
        }
    }
    return
}

mlp_checkpoint_init :: proc(checkpoint: ^MLP_Checkpoint, mlp: ^MLP, allocator := context.allocator) {
    checkpoint.weights = make([]f32, len(mlp.weights), allocator)
    checkpoint.biases = make([]f32, len(mlp.biases), allocator)
}

mlp_checkpoint_destroy :: proc(checkpoint: ^MLP_Checkpoint) {
    delete(checkpoint.weights)
    delete(checkpoint.biases)
}

mlp_checkpoint_save_to_json_file :: proc(checkpoint: ^MLP_Checkpoint, file_name: string, temp_allocator := context.temp_allocator) {
    marshalled, err := json.marshal(checkpoint^, allocator = temp_allocator)
    if err != nil || !os.write_entire_file(file_name, marshalled) {
        fmt.eprintfln("Failed to save %v", file_name)
    }
}

mlp_checkpoint_load_from_json_file :: proc(checkpoint: ^MLP_Checkpoint, file_name: string, temp_allocator := context.temp_allocator) {
    data, ok := os.read_entire_file_from_filename(file_name, temp_allocator)
    if !ok {
        fmt.eprintfln("Failed to load %v", file_name)
    }
    err := json.unmarshal(data, checkpoint, allocator = temp_allocator)
    if err != nil {
        fmt.eprintfln("Failed to save %v", file_name)
    }
}

mlp_save_checkpoint :: proc(mlp: ^MLP, checkpoint: ^MLP_Checkpoint) {
    copy(checkpoint.weights, mlp.weights)
    copy(checkpoint.biases, mlp.biases)
}

mlp_load_checkpoint :: proc(mlp: ^MLP, checkpoint: ^MLP_Checkpoint) {
    copy(mlp.weights, checkpoint.weights)
    copy(mlp.biases, checkpoint.biases)
}

//==========================================================================
// Data Set
//==========================================================================

Data_Set :: struct {
    size: int,
    input_size: int,
    output_size: int,
    input: []f32,
    output: []f32,
    order: []int,
}

data_set_init :: proc(data_set: ^Data_Set, size, input_size, output_size: int, allocator := context.allocator) {
    data_set.size = size
    data_set.input_size = input_size
    data_set.output_size = output_size
    data_set.input = make([]f32, size * input_size, allocator)
    data_set.output = make([]f32, size * output_size, allocator)
    data_set.order = make([]int, size, allocator)
    for i in 0 ..< len(data_set.order) {
        data_set.order[i] = i
    }
}

data_set_destroy :: proc(data_set: ^Data_Set) {
    delete(data_set.input)
    delete(data_set.output)
    delete(data_set.order)
}

data_set_shuffle :: proc(data_set: ^Data_Set) {
    rand.shuffle(data_set.order)
}

data_set_select :: proc(data_set: ^Data_Set, i: int) -> (input, output: []f32) {
    input = data_set.input[i * data_set.input_size:][:data_set.input_size]
    output = data_set.output[i * data_set.output_size:][:data_set.output_size]
    return
}

data_set_sample :: proc(data_set: ^Data_Set, i: int) -> (input, output: []f32) {
    i_in_order := data_set.order[i]
    input = data_set.input[i_in_order * data_set.input_size:][:data_set.input_size]
    output = data_set.output[i_in_order * data_set.output_size:][:data_set.output_size]
    return
}

data_set_load_csv :: proc(data_set: ^Data_Set, file_name: string, allocator := context.temp_allocator) -> (ok: bool) {
    file_data, success := os.read_entire_file(file_name)
    if !success {
        return
    }

    csv_reader: csv.Reader
    csv.reader_init_with_string(&csv_reader, cast(string)file_data, allocator)
    defer csv.reader_destroy(&csv_reader)

    _, _ = csv.read(&csv_reader)

    for i in 0 ..< data_set.size {
        values_str, err := csv.read(&csv_reader)
        if err != nil {
            break
        }

        y_int, _ := strconv.parse_i64(values_str[0])
        data_set.output[i * data_set.output_size + int(y_int)] = 1

        for j in 0 ..< data_set.input_size {
            value_int, _ := strconv.parse_i64(values_str[j + 1])
            data_set.input[i * data_set.input_size + j] = f32(value_int) / 255.0
        }
    }

    ok = true
    return
}

//==========================================================================
// Utility
//==========================================================================

leaky_relu :: proc(x: f32) -> f32 {
    return max(0.01 * x, x)
}

leaky_relu_derivative :: proc(x: f32) -> f32 {
    return x < 0 ? 0.01 : 1
}

relu :: proc(x: f32) -> f32 {
    return max(0, x)
}

relu_derivative :: proc(x: f32) -> f32 {
    return x <= 0 ? 0 : 1
}

sigmoid :: proc(x: f32) -> f32 {
    return 1.0 / (1.0 + math.exp(-x))
}

sigmoid_derivative :: proc(x: f32) -> f32 {
    return sigmoid(x) * (1.0 - sigmoid(x))
}

softmax :: proc(x: []f32) {
    max_value := slice.max(x)
    sum := f32(0)

    for i in 0 ..< len(x) {
        exp_value := math.exp(x[i] - max_value)
        x[i] = exp_value
        sum += exp_value
    }

    for i in 0 ..< len(x) {
        x[i] /= sum
    }
}

load_digit_png :: proc() -> (res: [INPUT_SIZE]f32) {
    digit_image, err := png.load_from_file("digit.png")
    if err != nil {
        fmt.eprintln("Failed to load digit.png")
    }
    pixels := mem.slice_data_cast([]image.RGBA_Pixel, digit_image.pixels.buf[:])
    for i in 0 ..< INPUT_SIZE {
        p := pixels[i]
        average_rgb := (f32(p.r) + f32(p.g) + f32(p.b)) / (255.0 * 3.0)
        res[i] = average_rgb * f32(p.a) / 255.0
    }
    return
}