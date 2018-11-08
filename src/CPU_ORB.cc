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
		double startime = (double)cv::getTickCount();//��ʼ��ʱ
		//cv:mat�Ĵ���ֻ�Ǵ�����ͷ����Ϣ������������Ϣ��ȫ����ͨ��������ֵ����ȡ������cv:mat��ȫ���Զ�̬��ʼ��
		rawimage = cv::imread(picname, 0);	//�ڶ���������ʾ����ͼ������ԣ�0��ʾ�Ե�ͨ�����룬1��ʾ����ͨ�����룬-1��ʾ��ͼ������ʽ���롣ORB��ȡ��Ҫ�Ҷ�ͼ��������0���롣
		nfeatures = mfeatures;
		nscaleFactor = mscaleFactor;
		nlevels = mlevels;
		//Ptr��opencv�е�һ������ָ�룻ORB��OPENCV�е�ORB�࣬����һ��ͼ��ORB�����ɣ���������Ҫָ��ORB��һЩ����������������������߶�ģ��������һ�����㡣һ�����ȱʡ��
		cv::Ptr<cv::ORB> orb = cv::ORB::create(mfeatures, mscaleFactor, mlevels);
		/*
		//ʹ��orb��������鿴ͼ�����Ҫ���ORB�����ӣ�����¼����
		orb->detect(rawimage, keypoints);
		//�����ɵ������Ӹ���ԭͼ�����������Ӿ���
		orb->compute(rawimage, keypoints, descriptors);
		*/
		orb->detectAndCompute(rawimage, cv::noArray(), keypoints, descriptors);
		//����ʱ����㣬����ӡ
		double costtime = (double)cv::getTickCount() - startime;//��������ʱ��=����ʱ��-��ʼʱ��
		printf("orb_cpu execution time = %gms\n", costtime*1000. / cv::getTickFrequency());//ת��ʱ�䵥λ�������������ʱ��
	}

	void ORB_CPU::MatchPic(ORB_CPU &orb1, ORB_CPU &orb2)
	{
		double startime = (double)cv::getTickCount();//��ʼ��ʱ
		//����������Ƚ�����Ȼ�󴴽��ȽϽ��������DMatch��opencv�е�һ�����ͣ����ڽ�����������Ƚϡ�
		cv::BFMatcher matcher(cv::NORM_HAMMING);
		std::vector<cv::DMatch> mathces;
		matcher.match(orb1.descriptors, orb2.descriptors, mathces);	//������ͼ���ƥ��������mathces�����У�����ÿ����Ա��¼����һ��ƥ���
		//��ƥ�������߽�����������ɺ���ͼ
		cv::Mat img_mathes;
		cv::drawMatches(orb1.rawimage, orb1.keypoints, orb2.rawimage, orb2.keypoints, mathces, img_mathes);
		//����ʱ����㣬����ӡ
		double costtime = (double)cv::getTickCount() - startime;//��������ʱ��=����ʱ��-��ʼʱ��
		printf("cpu match time = %gms\n", costtime*1000. / cv::getTickFrequency());//ת��ʱ�䵥λ�������������ʱ��
		//��ʾͼ��
		cv::imshow("CPU_ORB", img_mathes);
		cv::imwrite("CPU_ORB.jpg", img_mathes);
	}

	void ORB_CPU::waitKey(int times)
	{
		cv::waitKey(times);
	}
}


