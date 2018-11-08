#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "CPU_ORB.h"
namespace MORB
{
	using namespace std;
	ORB_CPU::ORB_CPU(const std::string& picname, int mfeatures, float mscaleFactor, int mlevels)
	{
		double startime = (double)cv::getTickCount();//开始计时
		//cv:mat的创建只是创建了头部信息，但是数据信息完全可以通过后续赋值来获取。所以cv:mat完全可以动态初始化
		rawimage = cv::imread(picname, 0);	//第二个参数表示读入图像的属性，0表示以单通道读入，1表示以三通道读入，-1表示按图像解码格式读入。ORB提取需要灰度图像，所以以0读入。
		nfeatures = mfeatures;
		nscaleFactor = mscaleFactor;
		nlevels = mlevels;
		//Ptr是opencv中的一种智能指针；ORB是OPENCV中的ORB类，用于一幅图像ORB的生成，我们这里要指定ORB的一些参数，例如特征点个数，尺度模糊比例，一共几层。一般可以缺省。
		cv::Ptr<cv::ORB> orb = cv::ORB::create(mfeatures, mscaleFactor, mlevels);
		/*
		//使用orb检测器，查看图像符合要求的ORB特征子，并记录坐标
		orb->detect(rawimage, keypoints);
		//将生成的特征子根据原图，生成描述子矩阵
		orb->compute(rawimage, keypoints, descriptors);
		*/
		orb->detectAndCompute(rawimage, cv::noArray(), keypoints, descriptors);
		//结束时间计算，并打印
		double costtime = (double)cv::getTickCount() - startime;//代码运行时间=结束时间-开始时间
		printf("orb_cpu execution time = %gms\n", costtime*1000. / cv::getTickFrequency());//转换时间单位并输出代码运行时间
	}

	void ORB_CPU::MatchPic(ORB_CPU &orb1, ORB_CPU &orb2)
	{
		double startime = (double)cv::getTickCount();//开始计时
		//创建特征点比较器，然后创建比较结果向量，DMatch是opencv中的一种类型，用于进行特征矩阵比较。
		cv::BFMatcher matcher(cv::NORM_HAMMING);
		std::vector<cv::DMatch> mathces;
		matcher.match(orb1.descriptors, orb2.descriptors, mathces);	//将两幅图像的匹配结果放入mathces向量中，向量每个成员记录了哪一对匹配点
		//将匹配结果用线进行了链接组成合体图
		cv::Mat img_mathes;
		cv::drawMatches(orb1.rawimage, orb1.keypoints, orb2.rawimage, orb2.keypoints, mathces, img_mathes);
		//结束时间计算，并打印
		double costtime = (double)cv::getTickCount() - startime;//代码运行时间=结束时间-开始时间
		printf("cpu match time = %gms\n", costtime*1000. / cv::getTickFrequency());//转换时间单位并输出代码运行时间
		//显示图像
		cv::imshow("CPU_ORB", img_mathes);
		cv::imwrite("CPU_ORB.jpg", img_mathes);
	}

	void ORB_CPU::waitKey(int times)
	{
		cv::waitKey(times);
	}
}


