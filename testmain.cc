#include "CPU_ORB.h"
#include "GPU_ORB.h"
#include <string>   
#include <iostream>
using namespace MORB;
int main()
{
	std::string picname1 = "005.jpg";
	std::string picname2 = "006.jpg";
	//建立两幅图像的ORB特征，每幅图各提取最多1000个特征点。
	ORB_CPU orb1(picname1, 1000);
	ORB_CPU orb2(picname2, 1000);

	//分别进行匹配
	//ORB_CPU::MatchPic(orb1, orb2);	//暴利匹配
	//ORB_CPU::MatchPic(orb1, orb2, true);	//重匹配
	ORB_CPU::MatchPic(orb1, orb2, false, true);	//滤波匹配

	ORB_GPU::waitKey(0);

	return 0;
}

