#ifndef __BRIEF_RI_H
#define __BRIEF_RI_H

#include <opencv2/opencv.hpp>

class BriefRIDescriptorExtractor: public cv::BriefDescriptorExtractor{
public:
    BriefRIDescriptorExtractor( float theta, int bytes = 32 );
    virtual void read( const cv::FileNode& );
	void setAngle( float theta ){
		theta_ = theta;
	}
    // virtual void write( FileStorage& ) const;

    // virtual int descriptorSize() const;
    // virtual int descriptorType() const;
private:
	float theta_;
protected:
    virtual void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    typedef void(*PixelTestWithPatternFn)(const cv::Mat&, const std::vector<cv::KeyPoint>&, cv::Mat&);
    PixelTestWithPatternFn test_pattern_fn_;

};

#endif
