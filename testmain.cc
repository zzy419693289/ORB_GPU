#include "CPU_ORB.h"
#include "GPU_ORB.h"
#include <string>   
#include <iostream>
using namespace MORB;
int main()
{
	std::string picname1 = "005.jpg";
	std::string picname2 = "006.jpg";
	//��������ͼ���ORB������ÿ��ͼ����ȡ���1000�������㡣
	ORB_CPU orb1(picname1, 1000);
	ORB_CPU orb2(picname2, 1000);

	//�ֱ����ƥ��
	//ORB_CPU::MatchPic(orb1, orb2);	//����ƥ��
	//ORB_CPU::MatchPic(orb1, orb2, true);	//��ƥ��
	ORB_CPU::MatchPic(orb1, orb2, false, true);	//�˲�ƥ��

	ORB_GPU::waitKey(0);

	return 0;
}

