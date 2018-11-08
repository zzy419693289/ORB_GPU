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
	//�������ʹ��GPUʵ��ORB���ӵ���ȡ������������
	//�����������Ӷ���һ����·�����ϴ�ͼ��GPU�ϣ�Ȼ�󴩼�GPU�汾�ĸ������࣬Ȼ��ִ��GPU�汾�����������ɣ�������
	ORB_GPU::ORB_GPU(const std::string& picname, int mfeatures, float mscaleFactor, int mlevels)
	{
		double startime = (double)cv::getTickCount();//��ʼ��ʱ
		//cv:mat�Ĵ���ֻ�Ǵ�����ͷ����Ϣ������������Ϣ��ȫ����ͨ��������ֵ����ȡ������cv:mat��ȫ���Զ�̬��ʼ��
		rawimage = cv::imread(picname, 0);	//�ڶ���������ʾ����ͼ������ԣ�0��ʾ�Ե�ͨ�����룬1��ʾ����ͨ�����룬-1��ʾ��ͼ������ʽ���롣ORB��ȡ��Ҫ�Ҷ�ͼ��������0���롣
		nfeatures = mfeatures;
		nscaleFactor = mscaleFactor;
		nlevels = mlevels;
		//Ҫʹ��GPU����ͼ�񣬵�һ���ǽ�ͼ���CPU�˴���GPU��
		gpuimage.upload(rawimage);
		//Ptr��opencv�е�һ������ָ�룻ORB��OPENCV�е�ORB�࣬����һ��ͼ��ORB�����ɣ���������Ҫָ��ORB��һЩ����������������������߶�ģ��������һ�����㡣һ�����ȱʡ��
		cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(mfeatures, mscaleFactor, mlevels);
		//ʹ��GPU�汾��orb��������鿴ͼ�����Ҫ���ORB�����ӣ�����¼���ꣻȻ�����������Ӿ���
		orb->detectAndComputeAsync(gpuimage, cv::cuda::GpuMat(), gpukeypoints, gpudescriptors);
		//����ʱ����㣬����ӡ
		double costtime = (double)cv::getTickCount() - startime;//��������ʱ��=����ʱ��-��ʼʱ��
		printf("orb_gpu execution time = %gms\n", costtime*1000. / cv::getTickFrequency());//ת��ʱ�䵥λ�������������ʱ��
		//��GPU�ϵ�������������Ӵ��ظ�CPU�ڴ�
		orb->convert(gpukeypoints, keypoints);
		gpudescriptors.download(descriptors);	//�����Ӷ��Ǿ���ֱ�Ӵ�GPU�����ؼ���
	}

	void ORB_GPU::GpuMatchPic(ORB_GPU &orb1, ORB_GPU &orb2)
	{
		double startime = (double)cv::getTickCount();//��ʼ��ʱ
		//����������Ƚ�����Ȼ�󴴽��ȽϽ��������DMatch��opencv�е�һ�����ͣ����ڽ�����������Ƚϡ�
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);	//orb���������Ƕ����������룬�����ú���������������������ѡL1��L2������HAMMING�Ⱦ��뷽ʽ���������Ǵ���GPU�汾�ıȽ�����ʹ��GPU�Ƚ�
		std::vector<cv::DMatch> mathces;
		matcher->match(orb1.gpudescriptors, orb2.gpudescriptors, mathces);	//������ͼ���ƥ��������mathces�����У�����ÿ����Ա��¼����һ��ƥ��㡣ע���������GPU�ϵ�2�������Ӿ���Ȼ��õ��Ľ������CPU�˵�ƥ��������
		//��ƥ�������߽�����������ɺ���ͼ
		cv::Mat img_mathes;
		cv::drawMatches(orb1.rawimage, orb1.keypoints, orb2.rawimage, orb2.keypoints, mathces, img_mathes);
		//����ʱ����㣬����ӡ
		double costtime = (double)cv::getTickCount() - startime;//��������ʱ��=����ʱ��-��ʼʱ��
		printf("gpu match time = %gms\n", costtime*1000. / cv::getTickFrequency());//ת��ʱ�䵥λ�������������ʱ��
		//��ʾͼ��
		cv::imshow("GPU_ORB", img_mathes);
		cv::imwrite("GPU_ORB.jpg", img_mathes);
	}

	void ORB_GPU::waitKey(int times)
	{
		cv::waitKey(times);
	}
}


