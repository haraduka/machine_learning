#include <iostream>
#include <fstream>
#include "../neural_network.hpp"
using namespace std;

/*
 * NMISTからのデータを取ってきて学習させてみた
 */

constexpr int dataSize = 784;

int main()
{
    ifstream ifs1("dataset/train-images.txt");
    ifstream ifs2("dataset/train-labels.txt");

    int dataNum = 300;
    vector< std::array<double, dataSize> > xs;
    vector<int> ys;
    for(int i=0; i<dataNum; i++){
        std::array<double, dataSize> dataset;
        int y;
        ifs2 >> y;
        for(int j=0; j<dataSize; j++){
            ifs1 >> dataset[j];
            dataset[j] /= 256;
        }
        xs.push_back(dataset);
        ys.push_back(y);
    }
    NeuralNetwork<dataSize, 300, 10> NN(4.0);
    NN.train(xs, ys, 300, 0.01);
    NN.saveWeight("nn1_weight.dat");

    cout << NN.predict(xs[0]) << " : right-> " << ys[0] << endl;
    cout << NN.predict(xs[1]) << " : right-> " << ys[1] << endl;
    cout << NN.predict(xs[2]) << " : right-> " << ys[2] << endl;
    cout << NN.predict(xs[3]) << " : right-> " << ys[3] << endl;
    cout << NN.predict(xs[4]) << " : right-> " << ys[4] << endl;
    cout << NN.predict(xs[5]) << " : right-> " << ys[5] << endl;
    cout << NN.predict(xs[6]) << " : right-> " << ys[6] << endl;
    cout << NN.predict(xs[7]) << " : right-> " << ys[7] << endl;
    cout << NN.predict(xs[8]) << " : right-> " << ys[8] << endl;
    cout << NN.predict(xs[9]) << " : right-> " << ys[9] << endl;
}
