#ifndef GPU_ORB_H
#define GPU_ORB_H

#include <opencv2/core/core.hpp> 
#include <opencv2/core/cuda.hpp> 
#include <string>

namespace MORB
{
	class ORB_GPU
	{
	public:
		ORB_GPU(const std::string& picname, int mfeatures = 500, float mscaleFactor = 1.2f, int mlevels = 8);	//输入变量缺省，只需要在声明的时候指明即可
		//注意我们在GPU上进行运算的时候，必须使用GPU的矩阵
		//图像原画
		cv::Mat rawimage;
		cv::cuda::GpuMat gpuimage;
		//KeyPoint是Opencv里面的一种类型，能够记录关键点的坐标等信息。我们这里的keypoints就是以向量的形式，增量式将该图中的所有关键点给记录下来
		//而在GPU上没有vector结构，所以也是以矩阵的形式进行记录关键点
		std::vector<cv::KeyPoint> keypoints;
		cv::cuda::GpuMat gpukeypoints;
		//descriptors是由原图像根据符合要求的关键点生成的每个关键点的描述子矩阵。其中矩阵的每一行代表一个关键点的描述子。以后主要就是根据描述子进行匹配比较
		//而gpu版本的descriptors是用GpuMat组成的，主要的区别是能够在GPU上进行运算
		cv::Mat descriptors;
		cv::cuda::GpuMat gpudescriptors;
		//根据两幅图像的ORB特性进行图像比较并显示。我们这里写成静态方法，方便调用。
		static void GpuMatchPic(ORB_GPU &orb1, ORB_GPU &orb2);	//注意两点，第一static关键词只需要在声明时指明即可；第二由于静态方法无法直接调用一般成员，这里只有通过对象引用来实现（因为静态方法出现于一般成员之前，所以这里不能定义形参，只有引用）
		static void waitKey(int times);
	private:
		//ORB参数
		int nfeatures;
		float nscaleFactor;
		int nlevels;
	};
}
#endif
