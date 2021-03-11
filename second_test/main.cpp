#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

int main(int, char**)
{
    /* Set up training data */
    /* 其中labels就表示是红球还蓝球 trainingData就表示每个球的坐标点 */
    int labels[4] = {1, 1, -1, -1};
    float trainingData[4][2] = { {401, 10}, {255, 10}, {501, 255}, {10, 501} };
    /* 把训练数据组建成SVM需要的格式Mat */
    Mat trainingDataMat(4, 2, CV_32F, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);

    /* Train the SVM */
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

    /* 准备可视化界面 */
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    /* 展示SVM结果给出的决策区域划分情况 */
    Vec3b green(0,255,0), blue(255,0,0);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            /* 准备predict函数的输入数据(是一个1行2列的flat类型的Mat) */
            Mat sampleMat = (Mat_<float>(1,2) << j,i);

            float response = svm->predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    }

    /* Show the training data */
    int thickness = -1;
    circle(image, Point(401, 10),  5, Scalar(0, 0, 0), thickness);
    circle(image, Point(255, 10),  5, Scalar(0, 0, 0), thickness);
    circle(image, Point(501, 255), 5, Scalar(0, 0, 0), thickness);
    circle(image, Point(10, 501),  5, Scalar(0, 0, 0), thickness);

    /* Show support vectors */
    thickness = 2;
    Mat sv = svm->getUncompressedSupportVectors();

    for (int i = 0; i < sv.rows; i++)
    {
        const float* v = sv.ptr<float>(i);
        circle(image, Point((int) v[0], (int) v[1]), 6, Scalar(0, 0, 255), thickness);
    }

    imwrite("result.png", image);

    imshow("SVM Simple Example", image);

    waitKey();
    return 0;
}