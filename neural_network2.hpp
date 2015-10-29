#pragma once
#include <iostream>
#include <fstream>
#include <array>
#include <random>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

//#define DEBUG

/*!
 * biasを入力層と隠れ層に追加したNeuralNetwork
 */

template<int n_in, int n_hid, int n_out>
class NeuralNetwork
{
private:
    /* それぞれの層に対するinputを計算する配列 */
    std::array<double, n_in+1>  input_in;
    std::array<double, n_hid+1> input_hid;
    std::array<double, n_out> input_out;

    /* それぞれの層に対するoutputを計算する配列 */
    std::array<double, n_in+1>  output_in;
    std::array<double, n_hid+1> output_hid;
    std::array<double, n_out> output_out;

    /* 隠れ層と出力層に対する誤差を計算する配列 */
    std::array<double, n_hid+1> error_hid;
    std::array<double, n_out> error_out;

    /* weight_'to'[from][to] で、fromからtoへの重みを格納 */
    std::array<std::array<double, n_hid+1> , n_in+1> weight_hid;
    std::array<std::array<double, n_out> , n_hid+1> weight_out;

    /* random */
    std::random_device randomDevice;
    std::mt19937 engine;

    /* sigmoid param */
    double beta = 1.0;

    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-beta*x));
    }

    double d_sigmoid(double x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    /* 使おうと思ったけど結局使っていない  */
    template<int N>
    void normalize(std::array<double,N>& x)
    {
        double mx = std::numeric_limits<double>::min();
        for(int i=0; i<N; i++){
            mx = std::max(mx, x[i]);
        }
        for(int i=0; i<N; i++){
            x[i] /= mx;
        }
    }

    void forwardPropagation(std::array<double, n_in>& x)
    {
        /* 入力はsigmoidに入れずにそのまま流す */
        for(int i=0; i<n_in; i++){
            input_in[i] = x[i];
            output_in[i] = input_in[i];
        }
        input_in[n_in] = 1;
        output_in[n_in] = input_in[n_in];

        /* 隠れ層は入力を重みづけしてsigmoidで発火 */
        for(int i=0; i<n_hid+1; i++){
            input_hid[i] = 0;
            for(int j=0; j<n_in+1; j++){
                input_hid[i] += weight_hid[j][i] * output_in[j];
            }
            input_hid[i] /= (n_in+1);
            output_hid[i] = sigmoid(input_hid[i]);
        }
        output_hid[n_hid] = 1;

        /* 出力層は重み付けして流す。*/
        for(int i=0; i<n_out; i++){
            input_out[i] = 0;
            for(int j=0; j<n_hid+1; j++){
                input_out[i] += weight_out[j][i] * output_hid[j];
            }
            input_out[i] /= (n_hid+1);
            output_out[i] = input_out[i];
        }
    }

    void backPropagation(double y, double rate_learn)
    {
        /* 出力層の誤差を計算 */
        for(int i=0; i<n_out; i++){
            if(i == y) error_out[i] = output_out[i] - 1.0;
            else error_out[i] = output_out[i];
        }

        /* 隠れ層から出力層への重みを更新 */
        for(int i=0; i<n_hid+1; i++){
            for(int j=0; j<n_out; j++){
                weight_out[i][j] -= rate_learn * error_out[j] * output_hid[i];
            }
        }

        /* 隠れ層の誤差を計算 */
        for(int i=0; i<n_hid+1; i++){
            error_hid[i] = 0;
            for(int j=0; j<n_out; j++){
                error_hid[i] += weight_out[i][j] * error_out[j];
            }
            error_hid[i] *= d_sigmoid(input_hid[i]);
        }

        /* 入力層から隠れ層への重みを更新 */
        for(int i=0; i<n_in+1; i++){
            for(int j=0; j<n_hid+1; j++){
                weight_hid[i][j] -= rate_learn * error_hid[j] * output_in[i];
            }
        }
    }

public:
    /* コンストラクタ(初期値はただのrandom値) */
    NeuralNetwork(double beta) : beta(beta)
    {
        /* randomエンジンの初期化 */
        engine.seed(randomDevice());
        std::uniform_real_distribution<double> Rand(-1.0, 1.0);

        /* weightは初期では0.0~1.0のrandom値 */
        for(int i=0; i<n_in+1; i++){
            for(int j=0; j<n_hid+1; j++){
                weight_hid[i][j] = Rand(engine);
            }
        }
        for(int i=0; i<n_hid+1; i++){
            for(int j=0; j<n_out; j++){
                weight_out[i][j] = Rand(engine);
            }
        }

    }

    /* コンストラクタ(初期値はfileから読み込む) */
    NeuralNetwork(const std::string& filename, double beta) : beta(beta)
    {
        /* randomエンジンの初期化 */
        engine.seed(randomDevice());
        std::uniform_real_distribution<double> Rand(-1.0, 1.0);

        std::ifstream ifs(filename);
        for(int i=0; i<n_in+1; i++){
            for(int j=0; j<n_hid+1; j++){
                ifs >> weight_hid[i][j];
            }
        }
        for(int i=0; i<n_hid+1; i++){
            for(int j=0; j<n_out; j++){
                ifs >> weight_out[i][j];
            }
        }

    }

    void saveWeight(const std::string& filename)
    {
        std::ofstream ofs(filename);
        for(int i=0; i<n_in+1; i++){
            for(int j=0; j<n_hid+1; j++){
                ofs << weight_hid[i][j] << " ";
            }
        }
        ofs << std::endl;
        for(int i=0; i<n_hid+1; i++){
            for(int j=0; j<n_out; j++){
                ofs << weight_out[i][j] << " ";
            }
        }
    }

    /*
     * xの次元は n_data * n_in
     * yの次元は n_data (ysのクラスタは0-indexedで、n_outとの兼ね合いがあるよね)
     * n_learnは施行回数
     * rate_learnは学習率
     */
    void train(std::vector<std::array<double, n_in>>& xs, std::vector<int>& ys, int n_learn = 100, double rate_learn = 0.1)
    {
        assert(xs.size() == ys.size());
        int n_data = static_cast<int>(xs.size());
        for(int n=0; n<n_learn; n++){
#ifdef DEBUG
            std::cout << n << " learn start" << std::endl;
#endif
            for(int i=0; i<n_data; i++){
                forwardPropagation(xs[i]);
                backPropagation(ys[i], rate_learn);
            }
        }
    }

    /*
     * data一つを受け取ってクラスタ番号を返す
     */
    int predict(std::array<double, n_in>& x)
    {
        forwardPropagation(x);
        double mx = std::numeric_limits<double>::min();
        int mx_index = -1;
        for(int i=0; i<n_out; i++){
#ifdef DEBUG
            std::cout << output_out[i] << " ";
#endif
            if(mx_index == -1 || output_out[i] > mx){
                mx = output_out[i];
                mx_index = i;
            }
        }
#ifdef DEBUG
        std::cout << std::endl;
#endif
        return mx_index;
    }
};
