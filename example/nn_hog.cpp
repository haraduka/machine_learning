#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include "../neural_network.hpp"
#include <opencv2/opencv.hpp>
using namespace std;

/*
 * NMISTからのデータを取ってきて学習させてみた
 * input noise rateで入れた分のrateのnoiseを画像に含ませることができる
 * hog記述子を今回は用いている。
 * opencvが必要です
 */

#define USE_NNHOG_WEIGHT

constexpr int dataSize = 784;
constexpr int inputSize = 441; //特徴量の次元
constexpr int imageSize = 28;
constexpr int N_HOG_SPLIT = 4; //これはimagesizeの約数でないとだめ(一つひとつのcellの大きさ)
constexpr int N_BIN = 9;
std::random_device randomDevice;
std::mt19937 engine;
std::string weight_filename = "nnhog_weight.dat";

vector<double> calculateHOG(const cv::Mat& img);

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

    /* ここからhogによる特徴量抽出 */
    vector< cv::Mat > images(dataNum);
    for(int i=0; i<dataNum; i++){
        images[i] = cv::Mat::zeros(imageSize, imageSize, CV_64F);
        for(int j=0; j<imageSize; j++){
            for(int k=0; k<imageSize; k++){
                images[i].at<double>(j, k) = xs[i][j*imageSize+k];
            }
        }
    }

    vector< std::array<double, inputSize> > xshog(dataNum);
    for(int i=0; i<dataNum; i++){
        auto tmp = calculateHOG(images[i]);
        for(int j=0; j<inputSize; j++) xshog[i][j] = tmp[j];
    }


#ifdef USEHOG_WEIGHT
    NeuralNetwork<inputSize, 200, 10> NN(weight_filename, 4.0);
#else
    NeuralNetwork<inputSize, 200, 10> NN(4.0);
    NN.train(xshog, ys, 500, 0.01);
#endif
    NN.saveWeight(weight_filename);

    //scoreの計算
    int tp[10], tp_fp[10], tp_fn[10];
    for(int i=0; i<10; i++){
        tp[i] = 0; tp_fp[i] = 0; tp_fn[i] = 0;
    }

    for(int i=0; i<dataNum; i++)
    {
        int pre = NN.predict(xshog[i]);
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

vector<double> calculateHOG(const cv::Mat& image) {
    cv::Mat xsobel, ysobel;
    cv::Sobel(image, xsobel, CV_64F, 1, 0);
    cv::Sobel(image, ysobel, CV_64F, 0, 1);

    vector<cv::Mat> bins1(N_BIN);
    for (int i = 0; i < N_BIN; i++){
        bins1[i] = cv::Mat::zeros(imageSize/N_HOG_SPLIT, imageSize/N_HOG_SPLIT, CV_64F);
    }

    //angleを求める
    cv::Mat Imag1, Iang1;
    cv::cartToPolar(xsobel, ysobel, Imag1, Iang1, true);
    add(Iang1, cv::Scalar(-180), Iang1, Iang1 >= 180);
    Iang1 /= (180/N_BIN);

    //histgram化
    for (int y = 0; y < image.rows-N_HOG_SPLIT; y+=N_HOG_SPLIT) {
        for (int x = 0; x < image.cols-N_HOG_SPLIT; x+=N_HOG_SPLIT) {
            for(int i=0; i<N_HOG_SPLIT; i++){
                for(int j=0; j<N_HOG_SPLIT; j++){
                    int ny = (y+i)/N_HOG_SPLIT;
                    int nx = (x+j)/N_HOG_SPLIT;
                    int ind1 = (int)Iang1.at<double>(y+i, x+j);
                    bins1[ind1].at<double>(ny, nx) += Imag1.at<double>(y+i, x+j);
                }
            }
        }
    }

    //正規化
    vector<double> res;
    for(int y=0; y<imageSize/N_HOG_SPLIT; y++){
        for(int x=0; x<imageSize/N_HOG_SPLIT; x++){
            double sum1 = 0;
            for(int i=0; i<N_BIN; i++){
                sum1 += bins1[i].at<double>(y, x);
            }

            if(sum1 == 0) sum1 = 1;
            for(int i=0; i<N_BIN; i++){
                res.push_back(bins1[i].at<double>(y, x)/sum1);
            }
        }
    }
    return res;
}
