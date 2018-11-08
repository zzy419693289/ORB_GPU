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
		ORB_GPU(const std::string& picname, int mfeatures = 500, float mscaleFactor = 1.2f, int mlevels = 8);	//�������ȱʡ��ֻ��Ҫ��������ʱ��ָ������
		//ע��������GPU�Ͻ��������ʱ�򣬱���ʹ��GPU�ľ���
		//ͼ��ԭ��
		cv::Mat rawimage;
		cv::cuda::GpuMat gpuimage;
		//KeyPoint��Opencv�����һ�����ͣ��ܹ���¼�ؼ�����������Ϣ�����������keypoints��������������ʽ������ʽ����ͼ�е����йؼ������¼����
		//����GPU��û��vector�ṹ������Ҳ���Ծ������ʽ���м�¼�ؼ���
		std::vector<cv::KeyPoint> keypoints;
		cv::cuda::GpuMat gpukeypoints;
		//descriptors����ԭͼ����ݷ���Ҫ��Ĺؼ������ɵ�ÿ���ؼ���������Ӿ������о����ÿһ�д���һ���ؼ���������ӡ��Ժ���Ҫ���Ǹ��������ӽ���ƥ��Ƚ�
		//��gpu�汾��descriptors����GpuMat��ɵģ���Ҫ���������ܹ���GPU�Ͻ�������
		cv::Mat descriptors;
		cv::cuda::GpuMat gpudescriptors;
		//��������ͼ���ORB���Խ���ͼ��Ƚϲ���ʾ����������д�ɾ�̬������������á�
		static void GpuMatchPic(ORB_GPU &orb1, ORB_GPU &orb2);	//ע�����㣬��һstatic�ؼ���ֻ��Ҫ������ʱָ�����ɣ��ڶ����ھ�̬�����޷�ֱ�ӵ���һ���Ա������ֻ��ͨ������������ʵ�֣���Ϊ��̬����������һ���Ա֮ǰ���������ﲻ�ܶ����βΣ�ֻ�����ã�
		static void waitKey(int times);
	private:
		//ORB����
		int nfeatures;
		float nscaleFactor;
		int nlevels;
	};
}
#endif
