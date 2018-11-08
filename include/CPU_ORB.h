#ifndef CPU_ORB_H
#define CPU_ORB_H

#include <opencv2/core/core.hpp> 
#include <string>

namespace MORB
{
	class ORB_CPU
	{
	public:
		ORB_CPU(const std::string& picname, int mfeatures = 500, float mscaleFactor = 1.2f, int mlevels = 8);	//�������ȱʡ��ֻ��Ҫ��������ʱ��ָ������

		//ͼ��ԭ��
		cv::Mat rawimage;
		//KeyPoint��Opencv�����һ�����ͣ��ܹ���¼�ؼ�����������Ϣ�����������keypoints��������������ʽ������ʽ����ͼ�е����йؼ������¼����
		std::vector<cv::KeyPoint> keypoints;
		//descriptors����ԭͼ����ݷ���Ҫ��Ĺؼ������ɵ�ÿ���ؼ���������Ӿ������о����ÿһ�д���һ���ؼ���������ӡ��Ժ���Ҫ���Ǹ��������ӽ���ƥ��Ƚ�
		cv::Mat descriptors;
		//��������ͼ���ORB���Խ���ͼ��Ƚϲ���ʾ����������д�ɾ�̬������������á�
		static void MatchPic(ORB_CPU &orb1, ORB_CPU &orb2);	//ע�����㣬��һstatic�ؼ���ֻ��Ҫ������ʱָ�����ɣ��ڶ����ھ�̬�����޷�ֱ�ӵ���һ���Ա������ֻ��ͨ������������ʵ�֣���Ϊ��̬����������һ���Ա֮ǰ���������ﲻ�ܶ����βΣ�ֻ�����ã�
		static void waitKey(int times);
	private:
		//ORB����
		int nfeatures;
		float nscaleFactor;
		int nlevels;
	};
}
#endif
