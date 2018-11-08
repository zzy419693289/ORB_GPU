#ifndef CPU_ORB_H
#define CPU_ORB_H

#include <opencv2/core/core.hpp> 
#include <string>

namespace MORB
{
	class ORB_CPU
	{
	public:
		ORB_CPU(const std::string& picname, int mfeatures = 500, float mscaleFactor = 1.2f, int mlevels = 8);	//输入变量缺省，只需要在声明的时候指明即可

		//图像原话
		cv::Mat rawimage;
		//KeyPoint是Opencv里面的一种类型，能够记录关键点的坐标等信息。我们这里的keypoints就是以向量的形式，增量式将该图中的所有关键点给记录下来
		std::vector<cv::KeyPoint> keypoints;
		//descriptors是由原图像根据符合要求的关键点生成的每个关键点的描述子矩阵。其中矩阵的每一行代表一个关键点的描述子。以后主要就是根据描述子进行匹配比较
		cv::Mat descriptors;
		//根据两幅图像的ORB特性进行图像比较并显示。我们这里写成静态方法，方便调用。
		static void MatchPic(ORB_CPU &orb1, ORB_CPU &orb2);	//注意两点，第一static关键词只需要在声明时指明即可；第二由于静态方法无法直接调用一般成员，这里只有通过对象引用来实现（因为静态方法出现于一般成员之前，所以这里不能定义形参，只有引用）
		static void waitKey(int times);
	private:
		//ORB参数
		int nfeatures;
		float nscaleFactor;
		int nlevels;
	};
}
#endif
