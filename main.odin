package main

import "core:os"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:strconv"
import "core:encoding/csv"
import "core:image"
import "core:image/png"

//
// Based on 3Blue1Brown's series on basic deep learning:
// https://www.youtube.com/watch?v=aircAruvnKk
//
// And the simple network code from:
// https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
//

MINI_BATCH_SIZE :: 10
INPUT_SIZE :: 784
HIDDEN_SIZE :: 30
OUTPUT_SIZE :: 10

main :: proc() {
    // train()
    predict_digit()
}

// Load digit.png at runtime and try to predict what digit it is.
// Comment this function out if you are going to train.
predict_digit :: proc() {
    // Load the raw model data at compiletime and bake it into the exe.
    network_data := #load("model.txt")
    network := (cast(^Network)&network_data[0])^

    digit_image, err := png.load_from_file("digit.png")
    if err != nil {
        fmt.eprintln("Failed to load digit.png")
    }

    pixels := mem.slice_data_cast([]image.RGBA_Pixel, digit_image.pixels.buf[:])

    sample: [INPUT_SIZE][1]f32
    for i in 0 ..< INPUT_SIZE {
        p := pixels[i]
        average_rgb := (f32(p.r) + f32(p.g) + f32(p.b)) / (255.0 * 3.0)
        sample[i][0] = average_rgb * f32(p.a) / 255.0
    }

    // See what the model thinks the number is.
    answer := argmax(network_infer(&network, sample))

    fmt.printfln("The digit is %v", answer)

    // Just to get the console to pause.
    buffer: [256]byte
    os.read(os.stdin, buffer[:])
}

// Train and validate with the mnist dataset.
train :: proc() {
    network: Network
    network_init(&network)
    network_train_from_csv_file(&network, "mnist_train.csv", "mnist_test.csv", context.temp_allocator)

    // It just dumps the raw model data into a text file.
    // This is probably a pretty bad way to do it.
    network_data := transmute([size_of(network)]byte)network
    if !os.write_entire_file("model.txt", network_data[:]) {
        fmt.eprintln("Failed to save model file")
    }

    free_all(context.temp_allocator)
}

//==========================================================================
// Linear Layer
//==========================================================================

Linear :: struct($I, $O: int) {
    weights: [O][I]f32,
    biases: [O][1]f32,
}

linear_init :: proc(linear: ^Linear($I, $O)) {
    for i in 0 ..< O {
        for j in 0 ..< I {
            linear.weights[i][j] = f32(rand.norm_float64())
        }
    }
    for i in 0 ..< O {
        linear.biases[i][0] = f32(rand.norm_float64())
    }
}

//==========================================================================
// Neural Network
//==========================================================================

Training_Sample :: struct {
    x: [INPUT_SIZE][1]f32,
    y: [OUTPUT_SIZE][1]f32,
}

Network :: struct {
    input: Linear(INPUT_SIZE, HIDDEN_SIZE),
    output: Linear(HIDDEN_SIZE, OUTPUT_SIZE),
}

network_init :: proc(network: ^Network) {
    linear_init(&network.input)
    linear_init(&network.output)
}

network_infer :: proc(network: ^Network, input: [INPUT_SIZE][1]f32) -> [OUTPUT_SIZE][1]f32 {
    activation0 := input
    z1 := matrix_multiply(network.input.weights, activation0) + network.input.biases
    activation1 := matrix_sigmoid(z1)
    z2 := matrix_multiply(network.output.weights, activation1) + network.output.biases
    return matrix_sigmoid(z2)
}

network_train_from_csv_file :: proc(
    network: ^Network,
    training_file_name: string,
    test_file_name: string,
    allocator := context.allocator,
) {
    training_data := make([dynamic]Training_Sample, allocator)
    if !load_training_data_csv(training_file_name, &training_data, allocator) {
        fmt.eprintfln("Failed to load training data: %v", training_file_name)
    }

    test_data := make([dynamic]Training_Sample, allocator)
    if !load_training_data_csv(test_file_name, &test_data, allocator) {
        fmt.eprintfln("Failed to load test data: %v", test_file_name)
    }

    network_stochastic_gradient_descent(network, training_data[:], test_data[:], 30, 3)
}

network_backprop :: proc(network: ^Network, sample: Training_Sample) -> (gradient: Network) {
    z1 := matrix_multiply(network.input.weights, sample.x) + network.input.biases
    activation1 := matrix_sigmoid(z1)

    z2 := matrix_multiply(network.output.weights, activation1) + network.output.biases
    activation2 := matrix_sigmoid(z2)

    cost_derivative := activation2 - sample.y

    output_delta := cost_derivative * matrix_sigmoid_derivative(z2)

    gradient.output.weights = matrix_multiply(output_delta, matrix_transpose(activation1))
    gradient.output.biases = output_delta

    input_delta := matrix_multiply(matrix_transpose(network.output.weights), output_delta) * matrix_sigmoid_derivative(z1)

    gradient.input.weights = matrix_multiply(input_delta, matrix_transpose(sample.x))
    gradient.input.biases = input_delta

    return
}

network_update_mini_batch :: proc(
    network: ^Network,
    mini_batch: []Training_Sample,
    learning_rate: f32,
) {
    gradient: Network

    for sample in mini_batch {
        delta_gradient := network_backprop(network, sample)
        gradient.input.weights += delta_gradient.input.weights
        gradient.input.biases += delta_gradient.input.biases
        gradient.output.weights += delta_gradient.output.weights
        gradient.output.biases += delta_gradient.output.biases
    }

    multiplier := learning_rate / f32(MINI_BATCH_SIZE)

    network.input.weights -= gradient.input.weights * multiplier
    network.input.biases -= gradient.input.biases * multiplier
    network.output.weights -= gradient.output.weights * multiplier
    network.output.biases -= gradient.output.biases * multiplier
}

network_stochastic_gradient_descent :: proc(
    network: ^Network,
    training_data: []Training_Sample,
    test_data: []Training_Sample,
    epochs: int,
    learning_rate: f32,
) {
    training_sample_count := len(training_data)
    test_sample_count := len(test_data)

    for i in 0 ..< epochs {
        rand.shuffle(training_data)

        for j := 0; j + MINI_BATCH_SIZE <= training_sample_count; j += MINI_BATCH_SIZE {
            batch: [MINI_BATCH_SIZE]Training_Sample
            for k in 0 ..< MINI_BATCH_SIZE {
                batch[k] = training_data[j + k]
            }
            network_update_mini_batch(network, batch[:], learning_rate)
        }

        correct_answer_count := 0
        for sample in test_data {
            if argmax(network_infer(network, sample.x)) == argmax(sample.y) {
                correct_answer_count += 1
            }
        }

        fmt.printfln("Epoch %v : %v / %v", i, correct_answer_count, test_sample_count)
    }
}

//==========================================================================
// Utility Functions
//==========================================================================

load_training_data_csv :: proc(
    file_name: string,
    samples: ^[dynamic]Training_Sample,
    allocator := context.allocator,
) -> (ok: bool) {
    data, success := os.read_entire_file(file_name)
    if !success {
        return false
    }

    csv_reader: csv.Reader
    csv.reader_init_with_string(&csv_reader, cast(string)data, allocator)
    defer csv.reader_destroy(&csv_reader)

    _, _ = csv.read(&csv_reader)
    for {
        values_str, err := csv.read(&csv_reader)
        if err != nil {
            break
        }

        sample: Training_Sample

        y_int, _ := strconv.parse_i64(values_str[0])
        sample.y[y_int] = 1

        for i in 0 ..< INPUT_SIZE {
            value_int, _ := strconv.parse_i64(values_str[i + 1])
            sample.x[i][0] = f32(value_int) / 255.0
        }

        append(samples, sample)
    }

    return true
}

matrix_multiply :: proc(a: [$M][$N]f32, b: [N][$P]f32) -> (res: [M][P]f32) {
    for i in 0 ..< M {
        for j in 0 ..< P {
            res[i][j] = 0
            for k in 0 ..< N {
                res[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    return
}

matrix_transpose :: proc(a: [$M][$N]f32) -> (res: [N][M]f32) {
    for i in 0 ..< M {
        for j in 0 ..< N {
            res[j][i] = a[i][j]
        }
    }
    return
}

sigmoid :: proc(x: f32) -> f32 {
    return 1.0 / (1.0 + math.exp(-x))
}

matrix_sigmoid :: proc(x: [$M][$N]f32) -> (res: [M][N]f32) {
    for i in 0 ..< M {
        for j in 0 ..< N {
            res[i][j] = sigmoid(x[i][j])
        }
    }
    return
}

sigmoid_derivative :: proc(x: f32) -> f32 {
    return sigmoid(x) * (1.0 - sigmoid(x))
}

matrix_sigmoid_derivative :: proc(x: [$M][$N]f32) -> (res: [M][N]f32) {
    for i in 0 ..< M {
        for j in 0 ..< N {
            res[i][j] = sigmoid_derivative(x[i][j])
        }
    }
    return
}

argmax :: proc(x: [$M][1]f32) -> (res: int) {
    highest := min(f32)
    for i in 0 ..< M {
        value := x[i][0]
        if value > highest {
            res = i
            highest = value
        }
    }
    return
}