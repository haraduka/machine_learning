#include <iostream>
#include <fstream>
#include <random>
#include "../neural_network2.hpp"
using namespace std;

/*
 * NMISTからのデータを取ってきて学習させてみた
 * input noise rateで入れた分のrateのnoiseを画像に含ませることができる
 */

#define USE_NN2_WEIGHT

constexpr int dataSize = 784;
std::random_device randomDevice;
std::mt19937 engine;
std::string weight_filename = "nn2_weight.dat";

int main()
{
    engine.seed(randomDevice());
    std::uniform_real_distribution<double> Rand(0.0, 1.0);
    ifstream ifs1("dataset/train-images.txt");
    ifstream ifs2("dataset/train-labels.txt");

    int dataNum = 300;
    cout << "input noise rate: ";
    double rate; cin >> rate;
    vector< std::array<double, dataSize> > xs;
    vector<int> ys;
    for(int i=0; i<dataNum; i++){
        std::array<double, dataSize> dataset;
        int y;
        ifs2 >> y;
        for(int j=0; j<dataSize; j++){
            ifs1 >> dataset[j];
            dataset[j] /= 256;
            if(Rand(engine) >= (100.0-rate)/100.0){
                dataset[j] = Rand(engine);
            }
        }
        xs.push_back(dataset);
        ys.push_back(y);
    }
#ifdef USE_NN2_WEIGHT
    NeuralNetwork<dataSize, 300, 10> NN(weight_filename, 4.0);
#else
    NeuralNetwork<dataSize, 300, 10> NN(4.0);
    NN.train(xs, ys, 300, 0.01);
#endif
    NN.saveWeight(weight_filename);

    //scoreの計算
    int tp[10], tp_fp[10], tp_fn[10];
    for(int i=0; i<10; i++){
        tp[i] = 0; tp_fp[i] = 0; tp_fn[i] = 0;
    }

    for(int i=0; i<dataNum; i++)
    {
        int pre = NN.predict(xs[i]);
        int res = ys[i];
        cout << pre << " " << res << endl;
        tp_fn[res]++;
        tp_fp[pre]++;
        if(pre == res){
            tp[res]++;
        }
    }
    cout << "------------------------------" << endl;
    cout << "digits\tprecision\trecall\tFvalue" << endl;
    cout.precision(6);
    cout.setf(ios::fixed);
    for(int i=0; i<10; i++){
        double recall = (double)tp[i]/tp_fn[i];
        double precision = (double)tp[i]/tp_fp[i];
        double F1 = 2*recall*precision/(recall + precision);
        cout << i << "\t" << precision << "\t" << recall << "\t" << F1 << endl;
    }
}
