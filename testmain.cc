#include "CPU_ORB.h"
#include "GPU_ORB.h"
#include <string>   
#include <iostream>
using namespace MORB;
int main()
{
	std::string picname1 = "001.jpg";
	std::string picname2 = "002.jpg";
	//��������ͼ���ORB������ÿ��ͼ����ȡ���1000�������㡣
	ORB_CPU orb1(picname1, 1000);
	ORB_CPU orb2(picname2, 1000);
	//Ȼ��ʹ��GPU��������ͼ���ORB����
	ORB_GPU orb3(picname1, 1000);
	ORB_GPU orb4(picname2, 1000);

	//�ֱ����ƥ��
	ORB_CPU::MatchPic(orb1, orb2);	//��̬��������
	ORB_GPU::GpuMatchPic(orb3, orb4);	

	ORB_GPU::waitKey(0);

	return 0;
}

