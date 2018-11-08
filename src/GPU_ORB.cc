#include <opencv2/core/core.hpp> 
#include <opencv2/core/cuda.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <string>
#include <iostream>
#include "GPU_ORB.h"
namespace MORB
{
	using namespace std;
	//这里的是使用GPU实现ORB算子的提取和描述子生成
	//几乎所有算子都是一样套路，先上传图像到GPU上，然后穿件GPU版本的该算子类，然后执行GPU版本的描述子生成，并下载
	ORB_GPU::ORB_GPU(const std::string& picname, int mfeatures, float mscaleFactor, int mlevels)
	{
		double startime = (double)cv::getTickCount();//开始计时
		//cv:mat的创建只是创建了头部信息，但是数据信息完全可以通过后续赋值来获取。所以cv:mat完全可以动态初始化
		rawimage = cv::imread(picname, 0);	//第二个参数表示读入图像的属性，0表示以单通道读入，1表示以三通道读入，-1表示按图像解码格式读入。ORB提取需要灰度图像，所以以0读入。
		nfeatures = mfeatures;
		nscaleFactor = mscaleFactor;
		nlevels = mlevels;
		//要使用GPU处理图像，第一步是将图像从CPU端传入GPU上
		gpuimage.upload(rawimage);
		//Ptr是opencv中的一种智能指针；ORB是OPENCV中的ORB类，用于一幅图像ORB的生成，我们这里要指定ORB的一些参数，例如特征点个数，尺度模糊比例，一共几层。一般可以缺省。
		cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(mfeatures, mscaleFactor, mlevels);
		//使用GPU版本的orb检测器，查看图像符合要求的ORB特征子，并记录坐标；然后生成描述子矩阵
		orb->detectAndComputeAsync(gpuimage, cv::cuda::GpuMat(), gpukeypoints, gpudescriptors);
		//结束时间计算，并打印
		double costtime = (double)cv::getTickCount() - startime;//代码运行时间=结束时间-开始时间
		printf("orb_gpu execution time = %gms\n", costtime*1000. / cv::getTickFrequency());//转换时间单位并输出代码运行时间
		//将GPU上的特征点和描述子传回给CPU内存
		orb->convert(gpukeypoints, keypoints);
		gpudescriptors.download(descriptors);	//描述子都是矩阵，直接从GPU上下载即可
	}

	void ORB_GPU::GpuMatchPic(ORB_GPU &orb1, ORB_GPU &orb2)
	{
		double startime = (double)cv::getTickCount();//开始计时
		//创建特征点比较器，然后创建比较结果向量，DMatch是opencv中的一种类型，用于进行特征矩阵比较。
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);	//orb算子由于是二进制描述码，所以用汉明距离进行描述。这里可选L1，L2，或者HAMMING等距离方式。这里我们创建GPU版本的比较器，使用GPU比较
		std::vector<cv::DMatch> mathces;
		matcher->match(orb1.gpudescriptors, orb2.gpudescriptors, mathces);	//将两幅图像的匹配结果放入mathces向量中，向量每个成员记录了哪一对匹配点。注意是输入的GPU上的2个描述子矩阵，然后得到的结果放在CPU端的匹配向量中
		//将匹配结果用线进行了链接组成合体图
		cv::Mat img_mathes;
		cv::drawMatches(orb1.rawimage, orb1.keypoints, orb2.rawimage, orb2.keypoints, mathces, img_mathes);
		//结束时间计算，并打印
		double costtime = (double)cv::getTickCount() - startime;//代码运行时间=结束时间-开始时间
		printf("gpu match time = %gms\n", costtime*1000. / cv::getTickFrequency());//转换时间单位并输出代码运行时间
		//显示图像
		cv::imshow("GPU_ORB", img_mathes);
		cv::imwrite("GPU_ORB.jpg", img_mathes);
	}

	void ORB_GPU::waitKey(int times)
	{
		cv::waitKey(times);
	}
}


