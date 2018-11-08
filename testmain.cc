#include "CPU_ORB.h"
#include "GPU_ORB.h"
#include <string>   
#include <iostream>
using namespace MORB;
int main()
{
	std::string picname1 = "001.jpg";
	std::string picname2 = "002.jpg";
	//建立两幅图像的ORB特征，每幅图各提取最多1000个特征点。
	ORB_CPU orb1(picname1, 1000);
	ORB_CPU orb2(picname2, 1000);
	//然后使用GPU建立两幅图像的ORB特征
	ORB_GPU orb3(picname1, 1000);
	ORB_GPU orb4(picname2, 1000);

	//分别进行匹配
	ORB_CPU::MatchPic(orb1, orb2);	//静态方法调用
	ORB_GPU::GpuMatchPic(orb3, orb4);	

	ORB_GPU::waitKey(0);

	return 0;
}

