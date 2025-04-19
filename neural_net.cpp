// Parallel Feedforward Neural Network with OpenMP
// Author: Anshika Gaur | HPC Project

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

const int INPUT_NODES = 11;
const int HIDDEN_NODES = 16;
const int OUTPUT_NODES = 1;
const double LEARNING_RATE = 0.01;
const int EPOCHS = 100;

// Activation function and its derivative
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

// Load dataset
void load_dataset(const string &filename, vector<vector<double>> &features, vector<double> &labels)
{
    ifstream file(filename);
    string line;

    getline(file, line); // Skip header

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream ss(line);
        string value;
        vector<double> row;
        int col = 0;

        while (getline(ss, value, ';'))
        {
            try
            {
                if (col < INPUT_NODES)
                {
                    row.push_back(stod(value));
                }
                else
                {
                    labels.push_back(stod(value) / 10.0); // Normalize label
                }
                col++;
            }
            catch (const invalid_argument &e)
            {
                cerr << "Skipping bad value: " << value << endl;
                row.clear();
                break;
            }
        }

        if (row.size() == INPUT_NODES)
        {
            features.push_back(row);
        }
    }
}

// Neural Net class
class NeuralNet
{
public:
    vector<vector<double>> weights_input_hidden;
    vector<vector<double>> weights_hidden_output;

    NeuralNet()
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);

        weights_input_hidden.resize(INPUT_NODES, vector<double>(HIDDEN_NODES));
        weights_hidden_output.resize(HIDDEN_NODES, vector<double>(OUTPUT_NODES));

        for (int i = 0; i < INPUT_NODES; ++i)
            for (int j = 0; j < HIDDEN_NODES; ++j)
                weights_input_hidden[i][j] = dis(gen);

        for (int i = 0; i < HIDDEN_NODES; ++i)
            for (int j = 0; j < OUTPUT_NODES; ++j)
                weights_hidden_output[i][j] = dis(gen);
    }

    double train(const vector<double> &input, double target)
    {
        vector<double> hidden_layer(HIDDEN_NODES);
        vector<double> output_layer(OUTPUT_NODES);

        for (int j = 0; j < HIDDEN_NODES; ++j)
        {
            for (int i = 0; i < INPUT_NODES; ++i)
                hidden_layer[j] += input[i] * weights_input_hidden[i][j];
            hidden_layer[j] = sigmoid(hidden_layer[j]);
        }

        for (int j = 0; j < OUTPUT_NODES; ++j)
        {
            for (int i = 0; i < HIDDEN_NODES; ++i)
                output_layer[j] += hidden_layer[i] * weights_hidden_output[i][j];
            output_layer[j] = sigmoid(output_layer[j]);
        }

        double error = target - output_layer[0];
        double output_delta = error * sigmoid_derivative(output_layer[0]);

        vector<double> hidden_deltas(HIDDEN_NODES);
        for (int i = 0; i < HIDDEN_NODES; ++i)
            hidden_deltas[i] = output_delta * weights_hidden_output[i][0] * sigmoid_derivative(hidden_layer[i]);

        for (int i = 0; i < HIDDEN_NODES; ++i)
            weights_hidden_output[i][0] += LEARNING_RATE * output_delta * hidden_layer[i];

        for (int i = 0; i < INPUT_NODES; ++i)
            for (int j = 0; j < HIDDEN_NODES; ++j)
                weights_input_hidden[i][j] += LEARNING_RATE * hidden_deltas[j] * input[i];

        return error * error;
    }
};

int main()
{
    vector<vector<double>> features;
    vector<double> labels;
    load_dataset("wine.csv", features, labels);

    NeuralNet nn;

    auto start = high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        double total_loss = 0.0;

#pragma omp parallel for reduction(+ : total_loss) // #pragma omp parallel for reduction(+:total_loss)

        for (int i = 0; i < features.size(); ++i)
        {
            double loss = nn.train(features[i], labels[i]);
            total_loss += loss;
        }

        cout << "Epoch " << epoch + 1 << " | Loss: " << total_loss / features.size() << endl;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "\nTraining completed in: " << duration.count() << " ms" << endl;

    return 0;
}