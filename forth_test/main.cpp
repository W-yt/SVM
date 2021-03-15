//添加使用到的头文件
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include "stdlib.h"
//声明命名空间
using namespace std;
using namespace cv;
using namespace cv::ml;

//!训练数据参数
const int sample_num_perclass = 40;     //训练每类图片数量
const int class_num = 2;                //训练类数
//!所有图片尺寸归一化
const int image_cols = 200;              //定义图片尺寸
const int image_rows = 200;              //定义图片尺寸
//!生成的训练文件保存位置
char SVMName[40] = "../SVM.xml";              //分类器的训练生成的名字,读取时也按照这个名字来
#define RW      1                       //0为读取现有的分类器,1表示重新训练一个分类器
//!程序入口
double Hu[7];       //存储得到的Hu矩阵
Moments mo;         //矩变量
cv::Size size = cv::Size(image_cols, image_rows);
int main(void)
{
#if RW
    //!读取训练数据
    Mat trainingData = Mat::zeros(sample_num_perclass*class_num, 7, CV_32FC1);          //填入图像的7个Hu矩
    Mat trainingLabel = Mat::zeros(sample_num_perclass*class_num, 1, CV_32SC1);
    char buf[50];                       //字符缓冲区
    for(int i=0;i<class_num;i++)        //不同了类的循环
    {
        for(int j=0;j<sample_num_perclass;j++)      //一个类中的图片数量
        {
            //!生成图片的路径(不同类的图片被放在了不同的文件夹下)
//            sprintf(buf, "../charSamples/%d/%d.png", i, j+1);
            String buf;
            buf = "../train/"+to_string(i)+"/"+to_string(j+1)+".jpg";
            //!读取
            Mat src = imread(buf, 0);
            //!重设尺寸（归一化）
            Mat reImg;
//            resize(src, reImg, size, CV_INTER_CUBIC);
            resize(src, reImg, size, CV_INTER_CUBIC);
            Mat canny;
            Canny(reImg, canny, 200, 120);
            //!求Hu矩
            mo = moments(canny);
            HuMoments(mo, Hu);
            //!将Hu矩填入训练数据集里
            float *dstPoi = trainingData.ptr<float>(i*sample_num_perclass+j);  //指向源的指针
            for(int r=0;r<7;r++)
                dstPoi[r] = (float)Hu[r];
            //!添加对该数据的分类标签
            int *labPoi = trainingLabel.ptr<int>(i*sample_num_perclass+j);
            labPoi[0] = i;
        }
    }
//    imwrite("../res.png", trainingData);

    //!创建SVM支持向量机并训练数据
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(0.01);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, trainingLabel);
    svm->save(SVMName);
#else
    //读取xml文件
    Ptr<SVM> svm=SVM::load<SVM>(SVMName);
#endif
    //!读取一副图片进行测试
    Mat temp = imread("../test/1/5.jpg", 0);
    Mat dst;
    resize(temp, dst, size, CV_INTER_CUBIC);
    Mat canny;
    Canny(dst, canny, 200, 120);
    mo = moments(canny);
    HuMoments(mo, Hu);
    Mat pre(1, 7, CV_32FC1);
    float *p = pre.ptr<float>(0);
    for(int i=0;i<7;i++)
        p[i] = Hu[i];
    float res = svm->predict(pre);
    cout << "detect result: " << res << endl;
    return 0;
}